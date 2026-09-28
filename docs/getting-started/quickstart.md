# 第一个可检查的实验

先运行一个合成任务，生成评测报告并读回轨迹。这个示例不启动 Minecraft，也不需要 Torch 或 GPU。

## 1. 安装并运行

在仓库根目录执行，Python 版本为 3.10–3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . gymnasium
minestudio doctor
python examples/v2/task_evaluation.py --output-dir runs/task-demo
```

示例使用计数器后端：每次按下 forward 增加一个物品，获得三个物品后成功。
两组种子分别运行一次，正常输出为：

```text
Synthetic demo: 2/2 completed; success rate 1.0
Verified T+1 observation readback. Report: runs/task-demo/<run_id>/evaluation.json
```

每局包含 3 个动作与 4 帧观测。示例已实际验证这些对齐关系；结果仅表示合成任务成功。

## 2. 查看产物 { #artifacts }

```text
runs/task-demo/<run_id>/
├── evaluation.json
├── <attempt_id>/
│   ├── manifest.json
│   ├── reset.json
│   ├── transitions.jsonl
│   └── observations/*.npy
└── videos/<attempt_id>.mp4    # 添加 --video 时生成
```

`evaluation.json` 包含任务、种子、预算、策略标识与每次尝试结果。使用 CLI 验证并概览一条轨迹：

```bash
minestudio trajectory runs/task-demo/<run_id>/<attempt_id>
```

请把尖括号替换为实际目录名。安装 FFmpeg 后，下面的命令额外生成可播放的 MP4：

```bash
python examples/v2/task_evaluation.py --output-dir runs/task-video --video
```

视频用于目视检查，无损 NPY 图像是轨迹数据的依据。

## 3. 理解策略边界

策略只需声明状态初始化与动作计算。每局的状态由采样器分别持有：

```python
from minestudio.actions import noop_action


class WalkPolicy:
    def initial_state(self) -> int:
        return 0

    def act(self, observation, state=None, *, deterministic=False):
        action = noop_action()
        action["buttons"]["forward"] = 1
        return action, (0 if state is None else state) + 1
```

公共动作表达具名按键和相机角度。模型内部的离散索引由策略自己的 codec 处理。
只想了解采样协议时，还可以运行仅依赖 NumPy 的 `python examples/v2/local_rollout.py`。

## 4. 接入 Minecraft

完成[引擎安装](installation.md#minecraft-setup)与[VPT 权重转换](../guides/policies.md#legacy-vpt)后：

```bash
minestudio evaluate --task collect_oak_log --policy /path/to/vpt-export \
  --output-dir runs/oak-log --seeds 42 43 --max-episode-steps 100 --device cpu --video
```

这是短预算的接入检查，100 步不保证能收集到木头。相同的评测/轨迹/视频路径已用真实引擎和 VPT 1x 完成 3 步检查，
以任务预算截断结束；未把它计为采木成功。运行版本与资源摘要见[阶段二验证记录](../architecture/stage2-validation.md)。

[继续：定义任务 →](../guides/tasks.md)
