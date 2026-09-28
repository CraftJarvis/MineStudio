# 阶段三：BC 闭环验证

**工程闭环通过，当前短训练模型的行为质量未通过验收。**
2026-09-28，在单张 H800 上完成真实 10xx 数据读取、VPT BC、断点恢复、导出重载和 Minecraft 采样。
验证 loss 下降，但新策略的按键和镜头抖动明显增加；不能将这次结果当成性能提升。
本次为 `v2.0.0` 开发分支增量，包版本仍为 `2.0.0a2`，未发布 a3。

## 实际范围与资源

- 环境：Python 3.12.3、Torch 2.14.0+cu130、CUDA 0 / NVIDIA H800 PCIe 80 GB。
- 数据：`CraftJarvis/minestudio-data-10xx-v110`，revision `cc50651e51e76435f3931285215cfaa1d95b9f8f`。
  全量 43 文件、168,308,670,178 字节已完成大小与上游摘要校验。
- 固定开发划分：16 / 4 / 4 条 train / validation / test 记录，按 12 位来源 ID 分组，不跨 split。
  这 24 条记录共 286,208 个保留帧，不是完整训练集或任务成功率基准。
- 主实验：每条训练记录均匀选择 8 个 32 帧窗口，共 128 个；验证集每条 4 个，共 16 个。
  batch size 2，AdamW，LR `1e-5`，FP32，梯度范数上限 1，60 次更新，共 3,840 个监督步骤。
  没有用 test loss 选择 checkpoint。
- 初始模型：已严格转换的 VPT foundation 1x，共 70,998,654 参数。
  权重 SHA-256：`475fbd0df655ad77c3e3f602d157f4273032bff8e6e82c3863a992f5b03753f9`。
  数据窗口长度 32；预训练配置保持 sequence_length=128、memory_size=256，缓存长度仍为 128。

每个训练窗口独立初始化空缓存，不将窗口首帧伪装成 episode 首帧。
这是明确的截断上下文近似；本次没有验证连续流 TBPTT 或 burn-in。

## 数值与恢复证据

| 检查 | 实测结果 |
| --- | --- |
| 新旧数据读取 | 跨 train/validation/test 共 12 个窗口，包含块间和尾部；v1 图像解码/resize、按键与相机编码逐项一致 |
| 小样本过拟合 | 单个 32 帧窗口、LR `1e-4`、40 步；NLL **10.7772 → 0.2597** |
| 独立验证窗口 | 主实验 NLL **12.8848 → 7.8920**；按有效监督步骤平均 |
| 参数确实更新 | backbone 参数差 L2 **0.9451**，策略头 **0.3179**；价值头参数差为 0 |
| 导出重载 | 权重、同一输入的 log probability 与 recurrent state 完全一致 |
| 新进程恢复 | step 30 的 checkpoint 在新进程继续至 60；与参考继续训练的权重、AdamW、RNG、采样顺序和游标完全一致 |
| FP32 显存 | 验证脚本峰值 allocated 约 **3.47 GiB**；包含脚本保留的对照模型，不是最大吞吐测量 |
| BF16 | 公共训练示例完成 3 次 H800 更新、验证和导出；未声称 BF16 与 FP32 逐位一致 |

额外数值测试检查了整段与逐帧执行、跨缓存边界、batch 内 episode reset、缺失标签和尾部 mask、有效步骤归约、
中性相机 mask、梯度到达 backbone/策略头、连续训练与恢复更新一致。价值头不参与损失，共享网络更新仍会改变价值预测。

数据对照验证的是保留 v1 的同下标约定，尚未证明与原始录制视频/动作的逐帧语义完全一致。
旧转换器支持帧过滤和片段拼接；当前 LMDB 没有完整保留这些时间映射。
因此 legacy window 标记 `timebase="legacy_rows"`，未知奖励、终止/截断、真实 episode 开始和末尾额外观测均不补造。
不能把保留行直接当成已验证的固定 20 Hz 单步动力学数据。

## 游戏轨迹与行为

原模型与 BC 导出分别在 seed=42、采样种子 13、600 步预算下运行一次。
CUDA 推理和 VirtualGL 实际 GPU 渲染均通过预检。相同种子不是游戏全程确定性的保证，单次对照也不能估计成功率。

| 指标 | 原始 VPT | 60 步 BC |
| --- | ---: | ---: |
| 完成的转移 / 录像帧 | 600 / 601 | 600 / 601 |
| 空白帧 / 连续重复帧 | 0 / 0 | 0 / 0 |
| 视频相邻 PTS | 0.05 秒 | 0.05 秒 |
| 超过 5 米的位置跳变 | 0 | 0 |
| 按键状态切换 / 秒¹ | 12.23 | **60.93** |
| 相邻动作的镜头增量差均值 | 1.25° | **3.85°** |
| 镜头增量差 P95 | 6.78° | **12.66°** |
| 运动路径长度 | 97.06 米 | 126.86 米 |
| 采集表现 | 2 块云杉原木 | 打掉蕨类，没有原木入包 |
| 采橡木任务达成 | 否 | 否 |

¹ 对所有按键的 0/1 变化求和；一个步骤可同时切换多个按键，因此可能超过 20 次/秒。

接触图检查显示原模型有持续砍树行为，BC 模型频繁跳跃、低头与转向。
录像帧序、T+1 存储链和播放时间连续，但动作平滑度下降。没有使用平滑后处理掩盖这个差异。

BC 非 GUI 帧的相机响应误差 P95 约 `1.43e-5°`，但动作索引 205（从 0 计数）到观测 206 有一处 **25.81°** 异常：
该观测的位置和旋转回到先前状态，游戏 tick 仍前进。尚未确认是服务端校正还是后端状态问题，需保留为引擎侧待排查项。
GUI 打开期间另有相机命令不改变游戏视角。两个 run 的游戏统计 tick 都有暂存/补齐现象，
分别有 26 / 16 个非单位增量，不能据此把录制帧解释为严格逐 tick 的状态快照。

## 使用与复现

公共入口和 manifest 示例见 [BC 指南](../guides/behavior-cloning.md)。

```bash
python examples/v2/train_bc.py dataset.json artifacts/vpt runs/bc \
  --device cuda:0 --steps 60 --sequence-length 32 --batch-size 2
```

本机完整实验目录为 `runs/stage3-bc/`（Git 忽略），包含：

- `dataset.json`：固定数据引用；`verify.py`：v1 对照、过拟合、主训练、导出和新进程恢复。
- `training-summary.json`、`overfit.json`、`legacy-parity.json`、`fresh-resume.json`：数值证据。
- `train-first/`、`train-continuation/`、`train-fresh-resume/`：配置、逐步指标、恢复 checkpoint 和模型。
- `final-check/`：补充来源元数据和输入校验后的完整复测；与初次实验模型的权重摘要相同。
- `record.py`、`game/runs.json`：实际渲染采样脚本与录像/轨迹路径。
- `analyze.py`、`analysis/metrics.json`、`analysis/*-contact.png`、`analysis/training-loss.png`：行为和损失图。

确定性恢复实验使用 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、禁用 TF32、禁用 cuDNN benchmark，并启用 deterministic algorithms。
恢复只承诺已验证环境里的优化器步边界，不能从任意半步或任意设备精确接续。

CPU **82 项**、CUDA **85 项**回归均通过，无跳过；Ruff、格式、Pyright、MkDocs strict、wheel/sdist 构建及 wheel 中的新模块导入通过。
这些是本机结果，尚未取得本轮远端 CI 执行结果。

## 后续重点

这次完成了工程验收，也明确暴露了策略退化。下一轮优先验证原始片段时间映射，加入更长的连续上下文或 burn-in，
再比较更小学习率、参考策略 KL 约束与基于行为的 checkpoint 选择。保持 loss 和多种子游戏行为同时评估，
不能只根据验证 NLL 下降决定发布。后续阶段已完成同步 PPO 的首次实现，见 [PPO 验证记录](ppo-validation.md)；分布式训练和其他策略家族仍待迁移。
