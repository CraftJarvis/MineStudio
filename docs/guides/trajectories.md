# 读取与恢复轨迹

原生轨迹格式 `minestudio.trajectory`、schema version 1 保存每次尝试的完整上下文与有效转移。
写入只保留最近一帧；当前 reader 会把整次尝试载入内存，适合检查和后续转换。

## 读回完整 episode

```python
from minestudio.data import TrajectoryReader

record = TrajectoryReader("runs/demo/<run_id>/<attempt_id>").read()
print(record.status, record.metadata, record.outcome)
if record.status == "completed":
    episode = record.as_episode()
    assert len(episode.observations) == len(episode.transitions) + 1
```

`as_episode()` 只接受有真实终止或截断边界的 completed 记录。
失败尝试可能包含首帧和部分有效转移，也可能在 reset 前就失败；不会补造最后一步。

## 文件与完整性

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 格式/版本、动作与观测格式、元数据、状态、结果、转移数量与摘要 |
| `reset.json` | 首帧引用与初始化信息 |
| `transitions.jsonl` | 按序追加的动作、奖励、结束标志、info 和下一帧引用 |
| `observations/00000000.npy` | 首帧；之后每步只增加一张 NPY 图像 |

读取时校验已记录的 SHA-256、时间顺序和 episode 边界。图像引用不能越出轨迹目录。
info 编码保留 NumPy dtype/shape、tuple 与整数映射键；不使用 pickle，不接受任意 Python 对象或 object 数组。
摘要可发现意外损坏，不构成对不可信文件来源的认证。

数据集窗口、mask 和旧 LMDB 转换仍属于后续阶段。本格式用于保存真实在线转移，
不会把历史未知奖励和边界自动填为零或已知终态。

## 处理未完成的写入

状态为 `recording` 或 `interrupted` 时，默认读取会拒绝。确认需要检查有效前缀后显式恢复：

```python
record = TrajectoryReader("runs/demo/<run_id>/<attempt_id>").read(recover_incomplete=True)
print(record.status, len(record.transitions))
```

也可以执行 `minestudio trajectory DIRECTORY --recover`。恢复只读取已经完整提交的 JSONL 行，
忽略末尾未写完的一行和没有对应转移的孤立帧；已提交帧的摘要不匹配仍然报错。
恢复不改写源文件，也不会把中断记录升级为 completed。

写入目录必须是新目录，避免无意覆盖已有实验。文件追加和原子 manifest 替换用于进程中断恢复，
当前未提供断电后的持久性保证，也未实现从最后一帧恢复游戏进程。

[返回：采样与记录器 →](rollout.md)
