# v2 alpha 产物格式

本页记录 `2.0.0a2` 及当前开发分支已实现的 schema 1。任务的 `version` 是任务作者的版本号；
每条规则自己的 `kind` 与整数 `version` 确定解释方式，当前规则版本均为 1。

## 原生轨迹 manifest

| 字段 | 类型 / 约束 |
| --- | --- |
| `format` / `schema_version` | 固定 `minestudio.trajectory` / `1` |
| `action_format` / `observation_format` | 固定 `minecraft-native-v1` / `rgb-uint8-hwc-v1` |
| `status` | `recording`、`interrupted`、`completed`、`failed`、`cancelled` |
| `metadata` | 下述带类型的 info 编码；记录器包含 context 和 MineStudio 版本 |
| `fps` | 正有限数，播放时间基准 |
| `num_transitions` | 非负整数，结束写入时更新 |
| `journal_sha256` | 结束写入时生成的 transitions.jsonl 摘要 |
| `reset_sha256` | 有 reset 时在结束写入时生成 |
| `outcome` | 正常结束尝试时保存的结果；中断可以缺失 |

`context` 包含 run_id、case_id、attempt_id、seed 和 metadata。评测 metadata 包含完整 TaskSpec、
suite 身份、repeat、策略身份、预算来源、原始预算和确定性选项。
`outcome` 包含 context、status、num_steps、total_reward、terminated、truncated、success、end_reason、error、error_type。

`reset.json` 是 `{"observation": FRAME, "info": ENCODED_INFO}`。
`FRAME` 包含相对于 attempt 目录的 `path` 与 `sha256`，目标是 uint8 RGB 的 HWC NPY 文件。

每行 transition 包含 `index`（从零连续递增）、`action`（buttons 字典与二元素 camera 列表）、
`reward`（有限数）、布尔 `terminated` / `truncated`、`info` 和 `next_observation` 帧引用。
上一行 next_observation 就是本步 observation，首步使用 reset 帧。
JSONL 完整行才表示提交；未提交图像不参与恢复。完整 episode 不能在边界之后继续追加转移。

## info 编码

由 `core.serialization` 定义封闭的值类型。基础 JSON 标量原样表示，复合值使用显式 type 标签：

- dict 编码为 key/value 对列表，避免整数键被 JSON 强制改成字符串。
- list 与 tuple 使用不同标签，保留原集合类型。
- ndarray 保存 dtype、shape 与 base64 字节；拒绝 object dtype，读取检查数据长度与形状。
- 不能序列化任意用户对象。顶层 info 必须是字符串键字典，内部映射可以包含整数键。

读取使用类型标记解码，不动态 import 类、不执行 pickle。未知标记、版本或摘要不匹配时拒绝读取。
原子替换 JSON 与 flush JSONL 提供进程中断恢复语义，不承诺 fsync 级断电恢复。

## 评测报告

| 字段 | 内容 |
| --- | --- |
| `format` / `schema_version` | `minestudio.evaluation` / `1` |
| `run_id` / `status` | 运行身份；`running`、`finished`、`aborted` |
| `minestudio_version` / `python_version` | 软件运行版本 |
| `suite` | id、version、原始 tasks、seeds、repeats |
| `config` | 完整 EvaluationConfig，output_dir 转为字符串 |
| `resolved_cases` | case_id、准确 seed、包含最终任务与策略标识的 metadata |
| `result` | 正常返回后写入 EvaluationResult，包含 cases 与全部 attempts |
| `error` | 异常中止时写入异常类型及消息 |

无完成项的 success_rate、mean_return、mean_steps 是 JSON null。
报告保存配置先于运行；原始异常继续抛出。磁盘失败可能导致报告停留在 running，因此不能只凭该字段判断进程仍存活。
细节见[指标口径](../guides/evaluation.md#metrics)。

## VPT 导出

本地导出以 `policy_type="vpt"` 与 `schema_version=1` 标识，保存 config、codec、processor 和 weights_sha256，
权重文件固定为 `model.safetensors`。旧权重转换还写入 provenance，包括原配置、源配置/权重摘要及移除前缀。
此格式用于推理，不是包含优化器和 RNG 的训练恢复 checkpoint。

视频单独保存 MP4，不参与无损轨迹校验。它包含 T+1 帧；编码 fps、原始采样频率和实际执行吞吐不能混为一谈。


## 数据集 manifest 与 BC checkpoint

`minestudio.dataset` / schema 1 保存固定 records。每条记录包括 episode_id、source_group、split、storage 和 frames。
原生 storage=trajectory 引用已完成的记录目录；legacy_lmdb 显式引用 image/action 的路径、episode_idx、长度与 chunk_size。
reader 校验旧 LMDB 的实际 episode 身份和长度，不修改快照。可选的 repository、revision 和摘要记录来源身份。
数据划分先于窗口；同来源组跨 split 直接报错。完整示例见 [BC 指南](../guides/behavior-cloning.md)。

`minestudio.bc-checkpoint` / schema 1 使用 PyTorch 的 weights_only 兼容字典：

| 字段 | 语义 |
| --- | --- |
| `identity` | 训练目标、数据/验证 fingerprint、模型配置、参数布局、processor 和 codec |
| `initial_policy` | 已验证的初始导出 manifest；直接构造模型或较早的开发 checkpoint 可以为 null |
| `step` | 已完成的 optimizer step；恢复目标 max_steps 必须更大 |
| `model` / `optimizer` | 严格加载的权重与 AdamW 状态 |
| `order` / `cursor` / `sampler_rng` | 当前打乱后的窗口顺序、游标和独立采样 RNG |
| `rng` | Python、NumPy、Torch CPU 和 CUDA RNG |
| `recurrent_state` / `scheduler` | 首版均为 null，分别对应独立窗口和恒定 LR |

checkpoint 只在优化器步边界原子替换，不承诺半步恢复或断电持久化。
恢复要求相同数据与目标；设备/软件确定性属于额外验证条件。
每次 train_bc 创建新的输出目录和 run_id，config.json 记录该身份；TrainingResult 指向配置、指标、checkpoint 与模型导出。
推理仍使用 VPT manifest + safetensors，训练 checkpoint 不冒充模型导出。


## PPO rollout 与 checkpoint

`minestudio.ppo-rollout` / schema 1 保存只含张量和基础类型的字典，可用 `torch.load(weights_only=True)` 读取。
每个 sequence 包含策略输入、原样保留的编码动作、真实 first_mask、旧 log probability / value、奖励、
terminated / truncated、下一观测价值、advantage / return、episode IDs 与初始缓存。
`policy_version` 和 `state_version` 标识该采样轮次；所有片段必须属于同一行为版本。
该审计格式不是原生 T+1 轨迹格式，不包含全部环境 info 或游戏快照。

`minestudio.ppo-checkpoint` / schema 1 保存：

| 字段 | 语义 |
| --- | --- |
| `identity` | 环境 ID、目标和预算、模型/参数布局、FP32 与缓存/价值尺度语义 |
| `initial_policy` | 初始 VPT 导出 manifest |
| `iteration` | 已完成的采样/更新轮次 |
| `optimizer_steps` / `environment_steps` | 累计优化器更新数 / 真实采样步数 |
| `next_episode` | 下一次 reset 使用的 episode 游标；seed 为 `(env_seed + next_episode) % 2**32` |
| `model` / `optimizer` | 模型和 AdamW 状态 |
| `sampler_rng` / `rng` | 序列打乱和 Python/NumPy/Torch CPU/CUDA RNG |
| `environment_state` / `recurrent_state` / `scheduler` | 首版均为 null：整轮边界保存，下轮重新 reset，恒定学习率 |

完整轮次结束后原子替换 checkpoint；失败的采样轮次不会成为训练更新。
恢复到新的输出目录，保留原记录。可以更改总轮次数与是否保存 rollout，其他配置需相同。
确定性合成环境支持精确恢复；Minecraft 没有世界状态恢复承诺。
