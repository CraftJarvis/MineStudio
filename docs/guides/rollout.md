# 采样与记录

`RolloutRunner` 组合环境工厂、策略工厂、预算和记录器。基础本地路径只依赖 NumPy。

## 配额与失败

`num_episodes` 是整个 run 的 case 数量；`num_envs` 是本地交错执行的最大环境槽位。
本地执行同步轮流推进环境，不意味着多进程并行。

每个 case 的环境种子由 `(seed, case_id)` 推导，与完成顺序和槽位数量无关。重试使用相同 case 与种子、新的 `attempt_id`。
也可以向 Runner 传入 `cases=(RolloutCase(...), ...)`，为每个 case 指定准确的 seed、独立环境工厂与元数据；
case ID 必须唯一，case 数必须与 `num_episodes` 相等。评测使用这一路径保持原始种子。
只有 `BackendError` 参与 `max_retries` 重试；配置和代码错误直接抛出。策略自己的随机数需要由策略工厂管理。

`RolloutResult` 包含逐次尝试结果，以及最终 `completed`、`failed`、`cancelled` case 数；三项相加等于全局配额。
例如第一次失败、重试成功，只算一个 completed case，但保留两条 attempt 记录。

## 时间与边界

每个有效 transition 包含 `observation`、`action`、`reward` 和 `next_observation`。
T 步完整 episode 有 T+1 个观测。预算截断保留真实末帧，并标记 `truncation_reason="rollout_budget"`。
崩溃前的数据作为部分尝试保留，失败时不创建额外转移。

这里的 `Episode` 仅表达完整在线轨迹。历史数据的未知 reward、边界和监督 mask 将在数据迁移阶段单独实现。

## 记录器与资源

记录器实现 `on_episode_start`、`on_reset`、`on_transition`、`on_episode_end`。
开始事件发生在 reset 之前，因此初始化失败也有 attempt 身份。回调获得快照；记录器自身异常应使运行失败。

工厂必须返回新资源，不能反复返回同一个活跃环境或策略。Runner 会关闭其环境，以及提供 `close()` 的策略。
使用 `with RolloutRunner(...)` 管理生命周期。调用 `cancel()` 后会在下一步边界停止，未开始的 case 也计入 cancelled。

`MemoryRecorder` 用于小实验；它保留完整图像、部分失败尝试和已完成 `Episode`。
`TrajectoryRecorder` 增量写入无损轨迹；`VideoRecorder` 通过 FFmpeg 写入 MP4。
持久化记录器由调用方创建并关闭，建议与 Runner 一起使用上下文管理：

```python
from minestudio.rollout import RolloutConfig, RolloutRunner, TrajectoryRecorder, VideoRecorder

# make_env 与 make_policy 每次创建新的环境/策略。
with (
    TrajectoryRecorder("runs/sample") as trajectories,
    VideoRecorder("runs/sample") as videos,
    RolloutRunner(
        make_env,
        make_policy,
        RolloutConfig(num_episodes=2, max_episode_steps=100),
        recorders=[trajectories, videos],
    ) as runner,
):
    result = runner.collect()
```

视频记录策略分辨率的 T+1 帧，包含首帧与末帧。奇数宽高通过边缘补齐满足编码要求，不缩放原图。
reset 前失败的尝试没有视频；已有画面的失败尝试会尽量完成视频封装。
编码未完成时保留 `.partial.mp4`，诊断信息在 `.ffmpeg.log`。

轨迹里的 `fps` 和视频编码帧率默认 20，表示记录的播放时间基准，不代表实测吞吐；
使用非默认控制频率时应显式配置。读取与恢复见[轨迹指南](trajectories.md)。Ray 执行仍待后续迁移。
