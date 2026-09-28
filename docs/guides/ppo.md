# 从在线采样到 PPO 更新

PPO 使用当前策略新采集的游戏交互。首版支持 VPT、单设备 FP32、多个本地环境的同步采样、
GAE、策略/价值裁剪、熵项、KL 早停，以及完整采样轮次结束后的 checkpoint。
安装 `pip install -e '.[train,envs,tasks]'`，并按[安装指南](../getting-started/installation.md)配置引擎。

## 运行一个短实验

```bash
python examples/v2/train_ppo.py artifacts/vpt runs/ppo \
  --task collect_oak_log --device cuda:0 \
  --iterations 2 --rollout-steps 256 --sequence-length 32 --save-rollouts
```

可以传入 BC 或其他已验证的 v2 VPT 导出。任务文件可替换内置任务；奖励和成功条件应匹配地图中的资源。
示例输出 `environment.json`，记录完整任务和 EnvConfig，其内容摘要作为环境身份写入训练配置。

Python API 使用工厂，训练器管理环境创建与关闭：

```python
from minestudio.envs import EnvConfig, MinecraftEnv
from minestudio.policies import VPTPolicy
from minestudio.tasks import load_task
from minestudio.training import PPOConfig, train_ppo

task = load_task("collect_oak_log")
result = train_ppo(
    lambda: VPTPolicy.from_pretrained("artifacts/vpt", device="cuda:0"),
    lambda: MinecraftEnv(EnvConfig(), task=task),
    config=PPOConfig(
        output_dir="runs/ppo",
        environment_id="collect-oak-log-v1-default-env",
        iterations=2,
        rollout_steps=256,
        sequence_length=32,
    ),
)
```

`environment_id` 是调用者声明的固定环境/任务身份。修改任务、奖励、初始化、引擎等条件时必须更换身份。
示例用任务配置摘要避免手工复用旧标识；checkpoint 无法检查任意 Python 工厂的全部隐藏行为。

## 一个采样轮次如何运行

1. 使用当前版本权重，为每个环境 reset，初始化该策略版本的空缓存。
2. 每个环境采集 `rollout_steps` 个动作；采样期间模型不更新。多个环境当前依次执行，不是 Ray 或并行进程。
3. 保留实际采样的编码动作、行为 log probability、价值、奖励、真实边界及缓存版本。
4. 根据最后观测计算 bootstrap 和 GAE，检查旧概率重算一致性，然后打乱**连续序列**做多轮更新。
5. 保存 checkpoint。下一轮在新权重下重新 reset 环境并重新采集。

`rollout_steps` 必须是 `sequence_length` 的整数倍。序列内部不打乱时间，遇到 episode reset 使用真实 `first_mask`。
不同序列的缓存长度可能不同，`minibatch_sequences` 通过逐序列梯度累积组成一次 optimizer step，按有效步数加权。

!!! note "当前 recurrent PPO 的边界"

    每个序列保留采样时的起始缓存，并在更新时 detach。后续 minibatch/epoch 更新仍以该行为缓存为条件，
    这是明确的 truncated-BPTT 近似，未对整个 episode 从头重新计算缓存。
    每个缓存都带行为策略版本；采样轮次之间重新 reset，避免将旧权重缓存继续带入新行为策略。
    每轮末尾未自然结束的 episode 被标记为 rollout 截断，并从实际最后观测 bootstrap。
    这一实现适合短实验；长任务需要更长采样预算，跨轮持续环境与缓存重建尚未实现。

模型采样和更新均保持 `eval()`，更新时单独启用 autograd。这会冻结 dropout、BatchNorm 和价值归一化统计，
避免仅仅切换模式就改变旧概率。`eval()` 不会冻结模型参数。
VPT 价值输出使用反归一化后的奖励尺度；首版价值损失直接在该尺度计算，不更新 EWMA 统计，不动态缩放目标。

## 概率、GAE 与目标

`act_with_stats()` 保存的 `policy_action` 是实际采样的编码动作。VPT 的环境动作转换有损，
例如关闭相机的按键编码会让不同 camera codes 执行相同环境动作；因此不能把环境动作重新编码后计算旧概率。

每轮更新前，`check_behavior()` 在原始序列起始缓存下重新计算 log probability 和价值。
默认允许 log probability 最大绝对误差 `5e-4`；价值误差使用 `max(1, abs(old_value))` 归一化后检查。
不一致会直接中止更新。`initial_ratio_error` 应接近零，即更新前的概率比应接近 1。

| 边界 | 下一状态价值 | GAE 向下一条记录递推 |
| --- | --- | --- |
| 普通步骤 | 下一观测的行为价值 | 是 |
| 真实 terminated | 0 | 否 |
| 时间限制 / rollout 截断 | reset **之前**最后观测的价值 | 否 |

若 terminated 与 truncated 同时为 true，按真实终止处理。环境必须返回真正的最后观测，不能隐式 auto-reset。
`value()` 只求值，不采样动作、不消耗动作采样 RNG，也不推进调用者缓存。

策略目标是标准 clipped surrogate；价值损失为 `0.5 * MSE`，可选对旧价值做裁剪并取更大的平方误差。
总目标为 `policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy`。
advantage 默认在整个 rollout 上统一标准化。所有项按有效步骤平均，记录近似 KL 和 clip fraction。
KL 早停在当前 minibatch **应用 optimizer step 之前**检查；超限会丢弃其梯度并结束本轮更新。
这是基于 minibatch 的早停规则，不保证后续任何样本的 KL 都严格低于阈值。

环境异常直接中止当前调用并关闭已创建的环境；失败步骤不会变成训练转移，也不会被伪造为终止样本。

## 日志、恢复与评测

训练目录包含 `config.json`、`metrics.jsonl`、`checkpoint.pt` 和 `policy/`。
启用 `save_rollouts` 时还会保存 `rollout-XXXXXX.pt`，其中是可使用 `torch.load(weights_only=True)` 读取的普通张量字典。
该文件是 PPO 审计数据，包含输入序列与概率/价值/GAE/缓存，不冒充原生 T+1 轨迹格式。

```bash
python examples/v2/train_ppo.py artifacts/vpt runs/ppo-resumed \
  --task collect_oak_log --iterations 4 --resume runs/ppo/checkpoint.pt
```

`iterations` 是恢复后要达到的总轮次。除总轮次、输出目录和是否保存 rollout 外，配置必须相同。
checkpoint 保存模型、AdamW、RNG、采样器 RNG、下一个 episode 的种子游标及累计步数。
**它没有 Minecraft 世界快照。** 恢复会创建新环境，并从下一个已保存的种子开始新 episode。
确定性合成环境已验证新进程恢复精确一致；Minecraft 验证的是训练进程可接续，不承诺游戏轨迹逐位一致。

推理继续使用 `VPTPolicy.from_pretrained(result.export)`；checkpoint 与推理导出保持独立。
模型选择需要结合奖励、固定种子任务评测和轨迹行为。短 PPO 更新成功不代表任务能力已提高。
本机 GPU 与行为证据见 [PPO 验证记录](../architecture/ppo-validation.md)。
