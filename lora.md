# 全局线性层 LoRA 与 LWC 联合优化需求文档

## 1. 背景与核心目标
在模型量化过程中，仅依靠 LWC（Learnable Weight Clipping）等调整裁剪边界的方式存在一定的表达能力上限。为了进一步突破极低比特（如 INT2/INT3）下的精度瓶颈，本需求提出在**所有需量化的线性层（Linear Layers）上引入 LoRA**。

**核心设计理念（Mergeable PTQ）**：
我们在训练阶段通过 LoRA 对由于量化带来的损失进行梯度修复；但在部署（推理）阶段，我们**不希望引入任何双分支计算开销**。因此，前向传播的顺序必须是先将 LoRA 残差合并到 Base 权重上构成“有效权重”，再让 LWC 去决定如何最优地裁剪和量化这份组合后的有效权重。

## 2. 数学抽象与计算顺序逻辑

**绝对遵循“先 LoRA，后 LWC”的正向传播顺序：**

1. **组合有效权重（Effective Weight）**：
   $$W_{eff} = W_{base} + \frac{\alpha}{r} (B \cdot A)$$
   *其中 $W_{base}$ 为冻结的 FP16 原始权重，$A、B$ 为可学习的低秩矩阵。*
   
2. **应用量化与裁剪（LWC & Fake Quant）**：
   LWC 参数（`upbound_factor`, `lowbound_factor`）根据上述 $W_{eff}$ 的分布进行裁剪边界的计算，并施加伪量化（Fake Quantization）：
   $$W_q = Q_{\theta_{lwc}}(W_{eff})$$

3. **线性层推断（Linear Forward）**：
   $$Y = X \cdot W_q^\top$$

> **为什么如此设计？**
> 因为 LWC 学习的本质是“怎么量化当前这份权重最合适”。只有让 LoRA 参与塑造真正的被量化对象 ($W_{eff}$)，LWC 才能适配最终将要部署的权重分布。在训练结束后，我们可以将 LoRA 永久合并入 $W_{base}$ 中，获得完全等价于标准量化模型的极其干净的结构，实现真正的零推理开销（Zero Inference Overhead）。

## 3. 命令行参数配置

为保证代码的灵活性，要求通过命令行参数控制所有 LoRA 相关行为：

- `--use_linear_lora` (Switch): 是否全局开启线性层 LoRA。（默认: `False`）
- `--linear_lora_r` (int): 全局线性层 LoRA 的秩（Rank）大小。（默认: `16`）
- `--linear_lora_alpha` (float): 全局线性层 LoRA 的放缩参数 alpha。（默认: `16.0`）
- `--linear_lora_lr` (float): 全局线性层 LoRA 可学习参数的单独学习率。（默认: `1e-4` 或与 LWC 对齐，建议和 LWC 分开配置组）
- `--linear_lora_target_modules` (str): 允许以逗号分隔的形式指定挂载 LoRA 的模块。例如 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。若空或为 `all` 则默认为所有 QuantLinear 层。

说明：

- 当前代码中已存在 Router Gate LoRA 参数 `lora_r`、`lora_alpha`、`gate_lora_lr`，该需求中的全局线性层 LoRA 必须与之拆分，避免命令行语义冲突。
- 本文档中的 LoRA 默认指“QuantLinear 上的全局线性层 LoRA”，不等同于当前仅作用于 router gate 的 LoRA。

## 4. 工程实现步骤说明

### 4.1. 伪量化前向的挂入 (Hook into Forward)
在计算图层面，不需要使用 HuggingFace `peft` 库那种复杂的外部包装。应修改 `QuantLinear` 及其对应的临时量化权重构造逻辑，使所有量化路径共享同一份“有效权重”定义。

实现要求：

- 初始化 `QuantLinear` 时若 `use_linear_lora=True`，则注册 `self.lora_A` 和 `self.lora_B` 两个 `nn.Parameter`，分别用 Kaiming 初始化（或 Normal）和 Zero 初始化。
- 增加统一的有效权重构造逻辑，例如 `get_effective_weight()`：
  ```python
  if self.use_linear_lora:
      delta_w = (self.lora_alpha / self.lora_r) * (self.lora_B @ self.lora_A)
      current_weight = self.weight + delta_w
  else:
      current_weight = self.weight
  ```

- `QuantLinear.forward()` 在非 `temp_weight` 路径下，应基于 `current_weight` 再送入 `weight_quantizer`。
- `smooth_and_quant_temporary()` 不能继续直接使用 `module.weight.detach()` 或 `module.weight` 构造 `temp_weight`，而应改为对 `current_weight` 做量化。
- `smooth_and_quant_inplace()` 也必须基于 `current_weight` 执行最终量化；否则训练态和导出态将不一致。

也就是说，LoRA 的接入点不能只修改 `forward()`，而必须同时修改所有 `temp_weight` 相关路径，保证以下三条路径行为一致：

- 训练态临时量化路径
- 常规 fake quant forward 路径
- 最终 inplace / export 路径

### 4.2. 优化器的挂入 (Optimizer Setup)
在准备 Block 训练的 Optimizer 时：
- 识别所有启用了 LoRA 的线性层模块。
- 收集这些模块下的 `lora_A` 和 `lora_B`，以独立 `param_group` 注入到 `torch.optim.AdamW` 中，使用通过 `--linear_lora_lr` 指定的独立学习率更新。

要求：

- 全局线性层 LoRA 与 LWC/LET 共同参与同一个 loss 的反向传播，属于“联合优化”。
- 前向中的先后顺序固定为“先 LoRA 形成 `W_eff`，再由 LWC/quantizer 量化 `W_eff`”。
- 这意味着 LoRA 并不是独立于 LWC 存在的补偿支路，而是直接参与被量化对象的塑形。

### 4.3. 部署前合并 (Merge and Export)
如果训练最终使用了 LoRA：
- 训练完成后，在执行真实打包权重的 `real_quant` 或 `save_pretrained` 之前，必须调用一个 `merge_lora()` 逻辑。
- 逻辑非常简单：`self.weight.data = self.weight.data + delta_w.detach()`
- 合并后，将 LoRA 参数指针删除或从 `state_dict` 剥离。此时，模型退化为包含了修复特性的标准量化模型，完美进入现有的 AutoGPTQ/MLC-LLM 推理管线。

说明：

- 本需求不强制要求支持训练中断后的 LoRA checkpoint/resume 扩展。
- 首版重点是训练流程、量化路径和导出路径的一致性。

## 5. 与后续 MoE Expert Self-Supervision 的兼容性

本需求应兼容后续将要实现的“每个专家自己监督自己”的训练方案。

兼容性约束：

- LoRA 首先作为底层能力接入 `QuantLinear`，优先保证对任意 QuantLinear 层都能工作。
- 后续 MoE Expert Self-Supervision 实现时，可在 expert 的 `gate_proj`、`up_proj`、`down_proj` 上直接复用该能力，而无需重新设计新的 LoRA 模块。
- 若按阶段实现，应先完成全局线性层 LoRA 与 LWC 的统一有效权重路径，再在此基础上实现 expert self-supervision。