# 阶段四：PPO 最小闭环验证

2026-09-28，开发分支完成了 VPT 同步 PPO 的首次实现与 H800 实测。
本次覆盖采样、旧概率重算、GAE、策略/价值更新、checkpoint、新进程恢复和推理导出。
这是算法工程闭环验证，尚未做长期训练或多种子任务能力评估。包版本仍为 `2.0.0a2`，未声明稳定版发布。

## 实现语义

- `ActorCriticEvaluation` 提供可求导的奖励尺度 value；`VPTPolicy.value()` 无采样地计算 bootstrap。
- 每轮采样固定权重，保留实际编码动作、旧概率/价值、episode ID、真实边界和行为缓存版本。
- `terminated` 不 bootstrap；时间或采样预算截断使用 reset 前最后观测的 value，但切断 GAE trace。
- 连续序列内求导；序列起点使用 detached 行为缓存。更新前检查概率比接近 1。
- 每轮更新后重新 reset 环境；checkpoint 仅在整轮边界保存。没有 Minecraft 世界快照恢复。
- FP32、恒定学习率、冻结 EWMA 价值统计，策略和梯度更新均用 eval 模式。多个环境在本地同步依次采样。
- clipped policy / value loss、全 rollout advantage 标准化、序列梯度累积、熵项、梯度裁剪和 minibatch KL 早停。

这不是全 episode 反向传播；更新期间也不会用新参数重建全部历史缓存。
跨轮持续世界、异步采样、Ray、多 GPU、混合精度和 scheduler 尚未实现。完整定义见 [PPO 指南](../guides/ppo.md)。

## 算法和恢复测试

| 检查 | 结果 |
| --- | --- |
| GAE 手算对照 | 普通步、真实终止、时间截断、同时 terminated/truncated、episode 隔离通过 |
| PPO 目标 | 正/负 advantage 的策略裁剪、奖励尺度 value clipping、mask 和梯度检查通过 |
| 行为概率 | 保存原始采样代码，跨缓存长度和 episode reset 后重算通过；篡改旧概率或缓存版本会阻止更新 |
| Bootstrap | 使用最后观测与推进后的缓存；不使用 reset 帧、不消耗动作 RNG |
| KL 早停 | 超限 minibatch 的梯度被丢弃，剩余 epoch 停止 |
| 参数与导出 | backbone、策略头、价值线性头确实更新；价值归一化统计不变；导出重载一致 |
| 环境失败 | 异常透传、环境关闭、没有虚构终止转移或已完成的 checkpoint |
| 新进程精确恢复 | 确定性合成环境 3 轮 / 48 步 / 12 次更新：模型、AdamW、RNG、种子游标、采样器和重新生成的 rollout 全部一致 |

CPU **89 项**、CUDA **94 项**完整回归通过，均无跳过。Ruff、格式、Pyright、MkDocs strict、wheel/sdist 构建与 wheel 中的 PPO API 导入检查通过。
CI 已加入 PPO 测试；上述结果来自本机，尚无本轮远端 CI 结果。

## 真实 Minecraft / H800

环境为 Python 3.12.3、Torch 2.14.0+cu130、NVIDIA H800 PCIe 80 GB，CUDA 0。
VirtualGL 的 NVIDIA renderer 与实际 Java GPU 进程均已检查。
奖励为六种原木的 `mine_block` 增量之和，初始化完成后建立基线，不把初始化操作算作奖励。

| 实验 | 采样 / 更新 | 结果 |
| --- | --- | --- |
| Foundation 1x | seed 42、43；2 × 256 步，8 次 optimizer step | 完成，采木奖励均为 0；仅作为运行与数值验证 |
| Foundation 新进程恢复 | 从第 2 轮 checkpoint 接续，seed 44 再采 256 步 | 累计 768 步 / 12 次更新；完成，奖励 0；不宣称世界逐位恢复 |
| RL / diamond 2x | seed 42；1,024 步，2 epochs，16 次更新 | 获得 **7 分真实采木奖励**，7 个正奖励步骤；全部更新与导出完成 |

三组均使用序列长度 32、每 minibatch 4 个序列、LR `1e-6`、gamma `0.99`、GAE lambda `0.95`、
policy/value clip `0.2`、value coefficient `0.5`、entropy coefficient `0`、target KL `0.02`。
模型的原始 memory / sequence 配置保持不变，缓存长度仍为 128。

### RL 2x 的直接证据

- 初始权重 SHA-256：`d6793de99d5cc0cbcfc4680164ab6c7ec94dd96b7db8c9101034bddf193193a3`。
- 更新前 log probability 最大误差 **3.50e-5**，概率比距离 1 的最大误差 **3.50e-5**。
- value 最大归一化重算误差 **1.34e-6**。
- 最后一次更新的 minibatch approx KL **0.001586**、clip fraction **0.0078125**；没有触发 KL 早停。
- backbone 参数差 L2 **0.03065**、策略头 **0.001822**、价值线性头 **0.0006305**，归一化统计差为 0。
- 导出前后权重、log probability、value 和 recurrent state 完全一致。
- 采样预算截断时最后观测 bootstrap value **4.3803**；从保存的 reward/value/边界重新计算 return 和 advantage 完全一致。
- **805 / 1,024** 个编码动作在“解码为环境动作再重新编码”后会改变，验证了直接保留采样代码的必要性。
- 验证脚本峰值 allocated 约 **7.40 GiB**，包含保留的模型对照对象；不是吞吐基准。

7 分奖励来自更新前采样的行为策略，证明真实奖励进入 PPO 计算链路，不表示这 16 次更新已经提升了采木能力。

## 导出模型的游戏行为

原始 RL 2x 和 PPO 更新后的导出各采样 600 步，环境 seed=42、采样种子 13。
评测任务为 600 步内采集 16 块云杉原木，两个 run 都未达这个阈值。

| 指标 | 原始 RL 2x | PPO 更新后 |
| --- | ---: | ---: |
| 转移 / 视频帧 | 600 / 601 | 600 / 601 |
| 空白帧 / 连续重复帧 | 0 / 0 | 0 / 0 |
| 游戏 tick 增量 | 全部为 1 | 全部为 1 |
| 视频帧间 PTS | 0.05 秒 | 0.05 秒 |
| 超过 5 米的位置跳变 | 0 | 0 |
| 砍掉云杉原木 | 6 | 5 |
| 合成云杉木板 | 16 | 16 |
| 按键切换 / 秒 | 3.13 | 4.43 |
| 镜头增量变化均值 / P95 | 0.89° / 4.95° | 1.37° / 7.81° |
| 非 GUI 相机响应误差 P95 | 1.95e-6° | 2.40e-6° |

接触图显示两者均有持续砍树和合成行为。PPO 模型的按键切换与镜头变化有所增大，但没有观察到录像不连续或动作解码错误。
同种子游戏不能保证全程严格确定性，且这里只做了一次短对照：它确认导出模型仍可执行可辨识的行为，不证明效果提升。
完整录像、接触图和数值见本机 `game/runs.json` 与 `analysis/metrics.json`。

## 本机复现与产物

公共示例为 `examples/v2/train_ppo.py`，可读取任意经过验证的本地 VPT 导出。
本机产物在 Git 忽略的 `runs/stage4-ppo/`：

- `verify_game.py`、`environment.json`、`task.json`：完整真实环境验证和任务身份。
- `train/`、`resume/`、`diamond/`：模型、checkpoint、配置、逐步日志与保存的 rollout。
- `train-summary.json`、`resume-summary.json`、`diamond-summary.json`：参数差、显存和导出一致性。
- `verify_toy.py`、`toy/resume-verification.json`：独立进程的确定性恢复证据。
- `audit_rollout.py`、`rollout-audit.json`：GAE 重算及动作编码审计。
- `record.py`、`game/runs.json`：导出前后策略的游戏行为检查。

确定性合成恢复使用 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、禁用 TF32/cuDNN benchmark，并启用 deterministic algorithms。
真实引擎和 VirtualGL 启动方法沿用 `runs/stage2-gpu/README.md`。
