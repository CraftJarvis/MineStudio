# a2 产物格式

本页记录 `2.0.0a2` 已实现的 schema 1。任务的 `version` 是任务作者的版本号；
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
