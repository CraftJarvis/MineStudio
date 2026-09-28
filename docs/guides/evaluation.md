# 固定种子评测

评测将任务、种子与重复次数展开为一组固定 case，复用采样器执行，并生成报告。
先运行[本地示例](../getting-started/quickstart.md)，再替换后端或策略。

## 组织评测集

下面的真实游戏示例需要已安装的引擎和已转换的 VPT 权重：

```python
from pathlib import Path

from minestudio.envs import MinecraftEnv
from minestudio.evaluation import EvaluationConfig, EvaluationSuite, evaluate
from minestudio.policies.vpt import VPTPolicy
from minestudio.tasks import load_task

# 此摘要应对应实际加载的导出 manifest。
import hashlib

export = Path("artifacts/vpt")
revision = hashlib.sha256((export / "manifest.json").read_bytes()).hexdigest()
result = evaluate(
    lambda: VPTPolicy.from_pretrained(export, device="cpu"),
    suite=EvaluationSuite(
        id="oak-log",
        version="1",
        tasks=(load_task("collect_oak_log"),),
        seeds=(42, 43),
        repeats=1,
    ),
    env_factory=lambda *, task: MinecraftEnv(task=task),
    config=EvaluationConfig(
        policy_id="local-vpt",
        policy_revision=revision,
        processor_revision="rgb-uint8-cv2-linear-v1",
        codec_revision="vpt-mu-law-v1",
        output_dir=Path("runs/evaluation"),
        max_episode_steps=100,
        record_video=True,
    ),
)
print(result.success_rate, result.report_path)
```

每个任务必须定义成功条件，预算必须有限。`env_factory` 接收解析后的 `task`，必须实际应用它；
评测器会核对 reset info 中的配置。覆盖步数预算时，报告同时保存原任务预算、覆盖来源和最终配置。

## 指标口径 { #metrics }

| 字段 | 含义 |
| --- | --- |
| `requested` | 任务数 × 种子数 × repeats |
| `completed` | 正常终止或预算截断的 case 数，包含任务未成功的情况 |
| `failed` | 后端故障且重试耗尽的 case 数 |
| `cancelled` | 未完成且被取消的 case 数，包含尚未启动的 case |
| `successes` | 已完成 case 中明确 `success=True` 的数量 |
| `success_rate` | successes / completed；没有完成项时为 `None`，JSON 中为 `null` |
| `mean_return` / `mean_steps` | 只在已完成 case 上取平均，没有完成项时为 `None` |

比较成功率时同时报告 completed/requested 与 failed。后端故障被排除在成功率分母之外，
因此只看成功率会掩盖运行失败。`cases` 是已启动 case 的最终尝试，`attempts` 保留所有重试；
未启动 case 仍列在报告的 `resolved_cases` 中。

相同 case 的重试沿用原种子，使用新的 attempt ID，绝不重复计入分母。
任务定义的失败条件属于正常结束的 case，其 success 为 false；与后端故障的 failed 不同。

## 复现信息与产物

报告在执行之前写入 `status="running"`、suite、解析后的 cases、策略标识和运行配置。
正常返回后写入 `finished` 与结果；代码、依赖或记录异常中止时尝试写入 `aborted` 与异常信息，随后继续抛出原异常。
进程被强制终止时报告可能仍是 running，此时按轨迹 manifest 检查各 attempt。

指定 `output_dir` 会自动创建轨迹记录器；`record_video=True` 同时开启 FFmpeg 视频。
路径结构见[快速入门](../getting-started/quickstart.md#artifacts)，恢复规则见[轨迹指南](trajectories.md)。

`deterministic=True` 是默认值，但自定义策略的 RNG 由调用方管理。策略 revision、processor 与 codec 标识是调用方的声明，
不是评测器自动鉴定的事实。CLI 从 VPT 导出中读取 processor/codec 和 manifest 摘要。
已验证 VPT 1x 在 CPU 上的权重转换数值一致性，以及真实游戏的短程评测。
底层 Minecraft 在固定种子下的完整轨迹确定性与采集成功率仍需更长的实验。
