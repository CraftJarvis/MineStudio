# 定义任务

任务把初始化、奖励和结束条件放在同一份可保存的配置中。每个环境拥有独立运行状态；
同一个不可变 `TaskSpec` 可以用于多次实验。

## 从内置任务开始

```python
from minestudio.envs import MinecraftEnv
from minestudio.tasks import load_task

task = load_task("collect_oak_log")
env = MinecraftEnv(task=task)
```

此时不会启动游戏；在 `reset()` 时才会创建后端。内置任务将时间设为白天、清除天气，
以初始化结束后的背包为基准，增加一个橡木原木即成功，预算为 1200 步。
实际采集成功率尚待真实游戏验证。

无需游戏即可查看解析后的配置：

```bash
minestudio task collect_oak_log
```

## 写一份自己的配置

下面是简化的 JSON 任务，可保存为 `collect.json`：

```json
{
  "id": "collect_two_logs",
  "version": "1",
  "instruction": "Collect two additional oak logs.",
  "initialization": [
    {"kind": "command", "version": 1, "command": "/time set day"}
  ],
  "rewards": [
    {
      "id": "logs",
      "metric": ["metrics", "inventory", "oak_log"],
      "reward_per_unit": 1.0,
      "mode": "increase",
      "missing": "zero"
    }
  ],
  "success": [
    {
      "metric": ["metrics", "inventory", "oak_log"],
      "operator": "ge",
      "threshold": 2,
      "relative": true,
      "missing": "zero"
    }
  ],
  "failure": [],
  "max_episode_steps": 1200
}
```

```python
from minestudio.tasks import TaskSpec

task = TaskSpec.from_json("collect.json")
assert task.to_dict()["id"] == "collect_two_logs"
```

YAML 使用相同字段，通过 `TaskSpec.from_yaml()` 读取，需要安装 `minestudio[tasks]`。
未知字段、重复奖励规则 ID、无效数值和未支持的初始化规则版本会报错。
`instruction` 是任务说明，当前最小策略协议不会自动把文本注入策略输入。

## 明确规则的含义

| 规则 | 行为 |
| --- | --- |
| 初始化 | 顺序执行命令；最后一次命令返回的观测作为 obs0，然后建立指标基线 |
| `mode="increase"` | 相邻两步指标的正增量乘以 `reward_per_unit`；负增量贡献零 |
| `mode="delta"` | 相邻两步的有符号增量乘以 `reward_per_unit` |
| `relative=true` | 结束条件比较当前值与初始化后基线的差值 |
| success / failure | 各组内任一条件满足即命中；同一步两组都命中时失败优先 |
| 任务结束 | 成功或失败令 `terminated=True`；只有预算耗尽则 `truncated=True` |

结束条件支持 `ge`、`le`、`gt`、`lt`、`eq`，在有效 step 之后检查，不在 reset 时结束任务。
奖励默认与后端奖励相加；`info["reward_components"]` 分别记录后端、任务总量和每条规则贡献。
`increase` 衡量背包净增量，丢弃后重新拾取也会再次产生正增量；它并不等同于唯一采集事件计数。

## 指标缺失与诊断

指标路径从 `info` 开始。MineRL 提供聚合后的 `metrics.inventory`，物品名移除 `minecraft:` 前缀；
其他原始统计位于 `metrics.stats`。自定义后端可提供自己的指标命名。

默认 `missing="error"` 会抛出 `TaskMetricError`。`missing="zero"` 只允许在存在的父映射中缺少最后一个键，
例如空背包里没有 `oak_log`；整个 `metrics.inventory` 缺失仍报错，避免把遥测故障误认为零物品。

`info` 还包含完整的 `task` 配置、`success`、`task_step` 和 `end_reason`。
初始化命令不计入策略动作步数，reset info 中的 `initialization_steps` 单独记录命令数。
自定义后端若接收带初始化命令的任务，必须实现 `TaskContext.execute_command()`。

[继续：固定种子评测 →](evaluation.md)
