# 从 v1 迁移

v2 当前开发版本为 `2.0.0a2`。源码已统一移入 `src/minestudio`，请先安装项目再运行脚本，不要依赖工作目录自动暴露包。

## 当前可迁移的接口

| 原接口 | 新接口 | 需要改变 |
| --- | --- | --- |
| `MinecraftSim` | `MinecraftEnv` | 使用原生动作；显式安装引擎；正确区分 terminated/truncated |
| 策略 `get_action()` | `Policy.act()` | 显式传入并接收 state；输出原生动作 |
| `MineGenerator` | `RolloutRunner` | 传入新资源工厂，配置全局配额和重试 |
| 隐式 VPT 动作转换 | `VPTActionCodec` | 明确区分环境动作与模型编码 |
| 任务初始化与奖励 callback | `TaskSpec` 与规则 | 用明确的指标路径声明增量奖励和成功/失败条件 |
| 采样 worker 内录像 | `TrajectoryRecorder` / `VideoRecorder` | 独立管理记录器，保留 T+1 帧和失败 attempt |
| 临时评测脚本 | `EvaluationSuite` + `evaluate` | 固定任务/种子，声明策略版本，使用明确成功率分母 |

v1 callback 不能直接传给新环境。现有初始化命令、数值指标奖励和结束条件可按[任务指南](../guides/tasks.md)改写；
复杂事件、GUI、通用 wrappers 和批推理仍需后续迁移，不应仅靠重命名调用。

## 兼容区域

`models`、`simulator`、`data`、`offline`、`online`、`inference`、`benchmark`、`utils`、`tutorials` 暂时保留在 `src/minestudio`。
此阶段允许新的 MineRL/VPT 适配器依赖指定的旧后端和网络组件，其他新领域不得反向依赖这些区域。

`models`、`simulator`、`data` 顶层改为延迟加载。旧入口仍保留，但基础安装不会安装全部历史依赖。
`.[legacy]` 是过渡依赖集合，不承诺所有旧脚本均已验证。原有 tests 根目录脚本保留为手动实验，pytest 只收集新的 unit/integration 目录。

## 数据与权重

现有数据格式没有在此 alpha 中批量重写。不要把未知边界或缺失 reward 默认为正常终止/零奖励。
新的带 mask 数据契约、LMDB 转换器、BC/PPO 数值对照属于后续阶段。

新增[原生轨迹 reader/writer](../guides/trajectories.md)用于在线交互记录，尚不能替代历史 LMDB 数据集。

VPT 支持[配置转换与严格 state-dict 导入](../guides/policies.md#legacy-vpt)，
保留旧 timesteps 对 attention cache 的影响，并输出带来源摘要的本地 manifest。
VPT foundation 1x 在 Linux CPU 上的真实权重数值对照已通过；范围见[验证记录](../architecture/stage2-validation.md)。
其他 VPT 变体、Hub 自动下载、GROOT/ROCKET/STEVE1 仍需单独迁移验证。
