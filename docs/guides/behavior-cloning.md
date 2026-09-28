# 从离线轨迹到 BC 策略

从本地 VPT 导出和固定数据划分开始，训练一个能重新载入 Minecraft 的策略。
当前实现使用单设备 PyTorch，支持原生轨迹、只读旧 LMDB、FP32 / CUDA BF16、验证集评估和断点恢复。
安装 `pip install -e '.[train]'`；这个 extra 不再要求 Lightning、Hydra 或 W&B，旧训练入口仍可安装 `.[legacy]`。

## 先固定数据，再生成窗口

`TrajectoryDataset` 接收 `minestudio.dataset` v1 JSON manifest。
下面是原生轨迹示例；路径相对于 manifest 的 `dataset_root`，缺省为 manifest 所在目录。

```json
{
  "format": "minestudio.dataset",
  "schema_version": 1,
  "dataset_root": "./recordings",
  "records": [
    {"episode_id": "session-a", "source_group": "source-a", "split": "train",
     "storage": "trajectory", "path": "session-a", "frames": 600},
    {"episode_id": "session-b", "source_group": "source-b", "split": "validation",
     "storage": "trajectory", "path": "session-b", "frames": 600}
  ]
}
```

每个目录由 `TrajectoryWriter` 生成且已完成。`frames` 是动作数量，原生记录另有最后一个观测。
相同 `source_group` 不能出现在不同 split；同源录制、切片或别名应共享来源组。
Dataset 不自行随机切分。需要完整性身份时，在 manifest 保留来源 revision、文件摘要或原生轨迹 manifest。
数据 fingerprint 绑定 manifest、split 和窗口配置，不代替下载阶段的内容校验；训练期间应保持源文件不变。

```python
from minestudio.data import TrajectoryDataset

with TrajectoryDataset("dataset.json", sequence_length=32, split="train") as dataset:
    window = dataset[0]
    print(window.episode_id, window.start_step)
    print(window.observations["image"].shape)  # [33, H, W, 3]
    print(window.valid_mask.sum())            # 最多 32 个真实步骤
```

`stride` 是窗口起点间距，默认等于长度；它不会隔帧采样。尾部补齐有独立 mask。
`TrajectorySubset(dataset, indices)` 选择固定窗口并保存独立 fingerprint，不关闭父 dataset。
原生 reader 当前按整条记录读取并缓存一条记录；长记录仍可能占用较多内存。

??? info "读取已下载的 v1 LMDB"

    记录设为 `storage: legacy_lmdb`，提供 `modalities.image` 和 `modalities.action`。
    每个引用包含 `path`（相对 dataset_root 的 data.mdb）、`episode`、`episode_idx`、`num_frames`、`chunk_size`。
    两种模态的身份和长度必须与记录一致。下载目录只读，不重写旧分块。

    读取采用 v1 的同下标图像和动作约定，保留原生按键和相机增量，再由策略 processor 编码。
    相机转换为公共动作契约要求的 float32；源 LMDB 中的原始数组不修改。
    不把 legacy record 冒充完整 episode：未知 reward / boundary / first mask 均为 false，
    最后一个额外观测缺失时，其 observation mask 为 false。
    原录制片段的隐藏拼接点无法从现有 LMDB 恢复，需另行验证原始来源。
    窗口的 `timebase="legacy_rows"` 表示保留行序，不能保证相邻行就是相隔一个游戏 tick。
    原生窗口使用 `timebase="environment_steps"`；两者不能混用于需要精确时间间隔的动力学目标。

## 运行短训练

```bash
python examples/v2/train_bc.py dataset.json artifacts/vpt runs/bc \
  --device cuda:0 --steps 60 --sequence-length 32 --batch-size 2
```

示例均匀选择每条训练记录最多 8 个窗口、每条验证记录最多 4 个窗口，适合工程验证。
每步输出 JSON loss、梯度范数、实际监督步数和所用窗口索引；验证 loss 是全部选定验证窗口的加权均值。
生产实验应显式扩大覆盖，并另外保留 test 集。

也可以直接组合公共 API：

```python
from minestudio.policies import VPTPolicy
from minestudio.training import BCConfig, train_bc

policy = VPTPolicy.from_pretrained("artifacts/vpt", device="cuda:0")
# train_dataset / validation_dataset 是前面打开的 Dataset 或 Subset。
result = train_bc(
    policy, train_dataset,
    validation_dataset=validation_dataset,
    config=BCConfig(output_dir="runs/bc", max_steps=60),
)
```

!!! note "当前的上下文与损失定义"

    首版训练器使用 `context_mode="independent_windows"`：每个窗口从空上下文开始，窗口内可求导。
    它不把窗口起点标记为真实 episode 起点，也不在窗口之间携带缓存；这是显式的截断上下文近似。
    VPT 的 `sequence_length` 配置保持预训练值，不跟着数据窗口长度修改。

    损失为有效监督步骤上 `-(buttons_weight × log P(buttons) + camera_weight × log P(camera))` 的均值。
    有效监督要求真实步骤、当前图像和动作标签均存在，不要求下一帧图像或奖励已知。
    默认两头权重为 1，监督中性相机动作。`mask_neutral_camera=True` 可以排除 camera=60 的相机头项；
    分母仍为有效步骤数。v1 的时间求和、batch 平均与此不同，学习率不能直接照搬。
    价值头不参与 BC 目标；共享网络更新仍可能改变价值预测。

`VPTPolicy.prepare_batch(windows)` 使用与推理相同的 RGB resize 和 codec。
`evaluate_actions(batch, state=None)` 返回可求导 `[B,T]` log probability、分头概率、entropy 和 next state。
高级调用者可以自行携带、detach 批量 state；该 state 与单流推理的 `VPTState` 分开。
缺失的内部图像需要分窗，不能将中间空帧隐式作为有效历史。

## 恢复和重载

训练目录包含：

| 文件 | 用途 |
| --- | --- |
| `config.json` | 目标、数据 fingerprint、模型/processor/codec 身份 |
| `metrics.jsonl` | 每步 loss、梯度范数、数据索引、验证结果 |
| `checkpoint.pt` | 模型、AdamW、RNG、采样顺序和游标；优化器步边界原子替换 |
| `policy/` | v2 manifest + safetensors 推理导出 |

```bash
python examples/v2/train_bc.py dataset.json artifacts/vpt runs/bc-resumed \
  --steps 120 --resume runs/bc/checkpoint.pt
```

`--steps` 是恢复后要达到的总步数。使用新输出目录，保留此前产物；数据、窗口、batch size、目标、精度和种子必须一致。
可以修改结束步数和评估/保存频率。恒定学习率，无 scheduler、梯度累积或待恢复的跨窗口状态。
精确数值恢复需要相同硬件、软件与确定性配置；本机 FP32 H800 对照见[验证记录](../architecture/bc-validation.md)。
训练器不关闭调用者的 dataset，并在结束后恢复策略原有的 train/eval 模式。

推理直接读取 `result.export`，无需加载优化器：

```python
policy = VPTPolicy.from_pretrained("runs/bc/policy", device="cuda:0")
action, state = policy.act(observation, deterministic=False)
```

小样本过拟合、验证 loss、导出一致性和真实游戏行为分别提供不同证据。
短训练通过工程验收，不等于已学会挖钻石或提高任务成功率。
