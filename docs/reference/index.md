# 公共 API

以下是当前 alpha 中实际存在的入口。完整设计蓝图见[架构规范](../architecture/v2.md)，尚未实现的类型不会作为占位类导出。

| 模块 | 公共对象 |
| --- | --- |
| `minestudio.core` | `Observation`、`MinecraftAction`、`ImageSize`、`Transition`、`Episode`、`TrajectoryWindow` 和异常类型 |
| `minestudio.actions` | `BUTTONS`、`noop_action`、`copy_action`、`validate_action`、`ActionCodec`、`VPTActionCodec` |
| `minestudio.envs` | `EnvConfig`、`Environment`、`EnvBackend`、`MinecraftEnv` |
| `minestudio.policies` | `Policy`、实验性 `VPTPolicy` |
| `minestudio.policies.vpt` | `VPTConfig`、实验性 `VPTPolicy` |
| `minestudio.policies.batches` | `PolicyBatch`、`ActionEvaluation`、`LikelihoodPolicy`、`ActorCriticEvaluation`、`ActorCriticPolicy`（需要 Torch） |
| `minestudio.tasks` | `TaskSpec`、`InitializationRule`、`RewardRule`、`EndRule`、`TaskRuntime`、`TaskContext`、`TaskMetricError`、`load_task` |
| `minestudio.data` / `minestudio.data.storage` | `TrajectoryWriter`、`TrajectoryReader`、`RecordedTrajectory` |
| `minestudio.data` | `TrajectoryDataset`、`TrajectorySubset` |
| `minestudio.training` | `BCConfig`、`TrainingResult`、`train_bc`、`PPOConfig`、`PPOTrainingResult`、`train_ppo`（当前适配 VPT） |
| `minestudio.rollout` | `RolloutCase`、`RolloutConfig`、`RolloutRunner`、`Recorder`、`MemoryRecorder`、`TrajectoryRecorder`、`VideoRecorder`、结果类型 |
| `minestudio.evaluation` | `EvaluationSuite`、`EvaluationConfig`、`EvaluationResult`、`evaluate` |

API 页面从源码静态提取，不需要启动 Minecraft 或加载模型权重。
