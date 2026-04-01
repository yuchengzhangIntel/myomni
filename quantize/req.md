***

# 需求文档：Block-wise Loss 评估与 Router 扩展模块

## 1. 背景与目标
在现有 MoE/Decoupled 量化训练主流程中，目前的自监督 Loss 逐渐演变为分解至每一个专家的 Loss 分别加权求和的方式。在此背景下，为了在更宏观的层面（Layer/Block 级）上追踪量化带来的整体特征偏移，并为未来增加“使用该 Loss 进行反向传播更新 Router 和 Attention 层”提供代码预留位，特制定本开发需求。

我们希望引入一个指定间隔（例如每逢 N 个 epoch）的评估阶段，评估由新建文件承载，防止原入口 `omniquant.py` 进一步臃肿。

## 2. 模块划分与接口要求

### 2.1 新建解耦文件 `quantize/block_evaluator.py`
- **目的**：专注于每个 Epoch 评估时期的前向推理评估计算，并接管可能到来的二次参数更新逻辑。
- **优点**：能够分离评测开销，隔离变量，不对已有的量化、初始化等环境代码构成侵入风险。
- **传参要求**：通过方法调用的形式接收外部环境的对象（包含 `layer`, `qlayer`, `args`, 输入张量批次 `quant_inps`, 当前基准对照特征 `fp_inps` 及可选的 `fp_inps_2`、控制屏蔽等信息）。全部应以张量引用的形式进行传参，不可强行发生 Deepcopy 以免消耗额外显存。

### 2.2 周期性调用策略整合
- 在原 `omniquant.py` （或对应主训练方法）内部定义参数接口（例如 `args.eval_interval = 5`）。
- 每次层级循环（Layer-wise）遍历特定 epoch 中（例如 `if epoch % args.eval_interval == 0`），触发外部调用。

## 3. 核心计算方法 (遵循 `3e7fbcf1b97f036a5901be0ccdb200ec276567d2` 逻辑)

需在该新模块内实现两种不同的 Block-wise 后计算逻辑。两种方法计算 Block 最终输出的 `quant_out` 后，都**必须严格使用历史 Git 指定版本**的方式求取偏差：使用 `loss = loss_func(fp_inps[...], quant_out)`，而且如果有配置 `args.aug_loss`，则同时增加 `fp_inps_2` 产生的 MSE 项。

### 方法一：学生自主推理 (Student-driven Routing)
- **描述**：即直接使用当前的量化层 `qlayer` 进行正常前向推断。
- **要求**：激活 `qlayer` 内的门控与路由机制，通过其产生的 Router Logits 自行选择专家执行模块内部运算，拿到 Block 的输出计算上述 MSE Loss。

### 方法二：教师强制注入 (Teacher Forcing)
- **描述**：即屏蔽在推断期间 `qlayer` 的自由专家选取概率，根据上级教师分配决策强制将 Token 指派给等价的子专家通路。
- **要求**：使用 FP16 教师层 `layer` （或其遗留缓存）产生的 `expert_idx`（专家路由结果），在量化生 `qlayer` 前馈时覆写路由分发过程，强制其在相同门控通道下计算，最终汇总得出输出，使用上述的历史方案计算 MSE Loss。

## 4. 未来拓展设计要求
- **预留接口桩（Stub API）**：当前版本新文件主要提供纯前馈测量并产生 Loss 供分析。代码要求在此方法结尾，必须包含未实现的函数如 `update_router_by_block_loss(loss...)` 等桩位。这样，开发者随时可以填入 `.backward()`, `optimizer.step()` 之类的人工启发式微调算法，而无需再次改动流程架构。
- **内存优化注意**：在此过程若仅作 Validation 统计，所有新写的测试管线须裹覆于 `with torch.no_grad():` 以防在长达几十层的测试中撑爆 VRAM（若为将来的扩展开启了需要梯度的开关，必须保证变量的 `clear_temp_variable()` 正常触发）。