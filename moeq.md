# OmniQuant MoE 解耦量化训练与改进方案需求文档

## 背景
针对 MoE 模型在原有 OmniQuant Block 级别联合训练中出现的“专家能力偏移（Expert Shift）”和“专家互相补偿导致的权重崩溃”等问题，现提出一套基于“组件解耦与多专家增强干预”的全新优化策略。

## 需求详情

### 1. Attention 层的解耦与联合量化
- **目标**：将 Attention 层与 MoE 层拆分，单独进行量化参数的最佳拟合。
- **机制**：
  - 监督信号设定在 Attention 的输出端（即 `o_proj` 矩阵的输出，而不是把 Q/K/V 拆开单独优化）。
  - 对于给定的 Block 输入，优化 Attention 内的所有量化参数（如 Q, K, V, O 的 LWC/LET），以最小化量化 Attention 输出与 FP16 Attention 输出之间的 MSE。
  - **冻结操作**：Attention 层的量化参数优化完成后，将其冻结，不再参与后续 MoE 层的更新。

### 2. MoE 层基于 Top-N 的扩展路由量化
- **目标**：在量化校准阶段，让更多的专家获得训练（特征覆盖），增强模型在量化噪声下的鲁棒性和次优专家的处理能力。
- **机制**：
  - 增加一个参数 `quant_routing_top_n`（例如设为 4 或 6）。
  - 前向传播时，**依据 FP16 Router 的概率分布（Logits）**，按概率从高到低选取前 N 个专家。
  - 强制将当前 Token 分发给这 N 个专家进行量化前向计算与状态保存。

约束：

- `quant_routing_top_n` 仅影响训练阶段的专家覆盖率，不改变最终推理阶段真实 routing top-k 的定义。
- 当 `quant_routing_top_n` 大于推理 top-k 时，新增专家仅用于训练覆盖和监督增强，不应直接改变部署时路由行为。

### 3. 专家层的独立 Loss 计算与概率加权
- **目标**：切断专家之间的依赖拉扯，保证每个专家只对自己的特征负责。
- **动态监督机制（On-the-fly Labeling 显存优化）**：
  - **背景**：直接将所有专家 FP16 输出常驻 GPU，会使单 Block 监督信号内存占用按 Top-N 近似线性放大。
  - **首选方案：Label Cache（CPU 缓存）**
    1. Label 产生时刻发生在 GPU 上。即先用 FP16 原始专家在 GPU 上计算出各专家 `down_proj` 之后的参考输出。
    2. 参考输出一经计算完成，即在 `torch.no_grad()` 下拷贝到 CPU 并缓存，不要求在 GPU 上长期驻留。
    3. 训练某个 batch / expert 时，仅将当前需要的那部分 label 子张量从 CPU 搬运回 GPU，再与量化输出计算 loss。
    4. 应按 expert 或按 batch 组织缓存，避免无意义地整块回传全部 label。
  - **备选方案：On-the-fly 双前向**
    1. **FP16 前向**：调用 `set_quant_state(weight_quant=False)`，在 `torch.no_grad()` 上下文中跑一遍数据，拦截（Hook）并短暂保存各专家 `down_proj` 之后的 FP16 参考输出。
    2. **量化前向**：调用 `set_quant_state(weight_quant=True)`，开启梯度图再跑一遍数据，拦截各专家的量化后输出。
    3. **即时计算**：计算出 Loss 后立刻反向传播，随后释放该 Batch 拦截的参考特征。
  - **本需求首版建议采用 Label Cache（CPU 缓存）方案**，On-the-fly 双前向作为备选验证路径。
- **机制**：
  - 不再把各专家的输出进行加权求和并与 FP16 加权求和算总 MSE，而是**每个专家单独计算 Loss**：`Loss_expert_i = MSE(FP16_expert_i_out, Quant_expert_i_out)`。
  - 允许并行更新：将当前批次内的所有专家的独立 Loss 累加在一起 `Total_Loss = sum(Loss_expert_i)` 执行 Backward 更新参数。
  - **新增控制参数 `use_router_weight_in_loss` (bool)**：
    - 若设定为 `True`：在计算某个 Token 在其对应专家的 Loss 时，将其乘以 Router 给该专家的分配权重（即 `Loss = router_prob * MSE(...)`）。置信度低的分配将产生较少的更新干扰。
    - 若设定为 `False`：纯粹计算特征映射的 MSE，即认为“只要 Token 分给了这个专家，它的重要性就等同，要求专家同等学好”。

  补充要求：

  - 首版监督位置定义在每个 expert 自身的 `down_proj` 输出。
  - expert 内部的可训练参数范围可覆盖 `gate_proj`、`up_proj`、`down_proj`，但监督锚点仍放在 `down_proj` 之后。
  - 若后续接入全局线性层 LoRA，则 expert 中的 `gate_proj`、`up_proj`、`down_proj` 应复用统一的 `QuantLinear + LWC + LoRA` 逻辑，而不是引入新的 expert 专用 LoRA 实现。

  ## 4. 与全局线性层 LoRA 文档的兼容性

  本需求必须兼容“先实现 LoRA，再实现 expert self-supervision”的开发顺序。

  兼容性原则：

  - LoRA 先作为 `QuantLinear` 的底层能力实现，并统一接入 `temp_weight` / fake quant / inplace quant 路径。
  - Expert Self-Supervision 在此基础上实现，不重复定义新的线性层量化前向逻辑。
  - 若 expert 层启用了 LoRA，则量化前向顺序仍为：先 LoRA 形成有效权重，再由 LWC/quantizer 对该有效权重执行量化。
  - `use_router_weight_in_loss` 只影响 loss 加权，不影响 LoRA 与 LWC 的计算顺序。

  ## 5. Label Cache 工程约束

  为控制显存和带宽，首版实现建议采用“GPU 计算、CPU 存储、按需回传”的 label cache 设计。

  要求如下：

  - FP16 label 生成时位于 GPU。
  - label 缓存常驻 CPU，可使用 pinned memory 优化搬运。
  - 计算 loss 时，允许只搬运某个 batch、某个 expert 对应的局部子张量到 GPU，而不是回传整个 label tensor。
  - 若缓存布局允许，优先按 expert 分桶存储，便于 expert self-supervision 阶段顺序读取。

  说明：

  - 对于校准集中约 260000 token、hidden size 2048、Top-N 为 6 的场景，FP16 label 总量约为 6GB 级别，放在 CPU 上是可接受的首选方案。
  - 若后续验证发现 CPU-GPU 搬运成为瓶颈，再考虑切换到局部 on-the-fly 双前向。

---