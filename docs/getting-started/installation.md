# 安装与诊断

## 选择需要的功能

当前 alpha 从源码安装，尚未发布到 PyPI。Python 第一方代码兼容 3.10–3.12；游戏运行时的支持情况见[实现进度](../architecture/progress.md)。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
minestudio doctor
```

基础安装只需要 NumPy，包含任务 JSON、原生轨迹读写与评测编排。
`doctor` 输出版本、可选模块、引擎路径、Java、DISPLAY、Xvfb 和 FFmpeg 的检查结果，不启动游戏或下载文件。

| 安装项 | 用途 |
| --- | --- |
| `.[envs]` | Gymnasium 环境、MineRL 适配器与图像缩放 |
| `.[policies]` | 实验性 VPT 推理、Torch 与本地权重导出 |
| `.[tasks]` | YAML 任务配置；JSON 和内置任务无需此 extra |
| `.[dev]` | pytest、Ruff、Pyright、构建工具 |
| `.[docs]` | MkDocs Material 文档预览 |
| `.[legacy]` | 尝试运行旧数据/训练/模型代码所需的历史依赖集合；不代表已验证兼容 |

例如：

```bash
python -m pip install -e '.[envs,policies,dev]'
```

`data`、`train`、`distributed`、`gui` extras 也已拆分，但对应的 v2 公共接口尚未全部实现。
原生轨迹存储不需要 `data` extra；它目前用于旧 LMDB/视频数据依赖。合成任务示例只需基础包加 Gymnasium。
Torch 的 CPU/CUDA 版本请按本机需要从 PyTorch 官方源安装。

## 可选的视频与文档

`VideoRecorder` 调用系统 FFmpeg，不需要 PyAV；视频测试还使用 ffprobe。可用以下命令检查：

```bash
ffmpeg -version
ffprobe -version
python -m pip install -e '.[docs]'
mkdocs serve
```

文档采用本地系统字体、搜索、明暗主题与移动端导航，预览默认位于 `http://127.0.0.1:8000`。
文档已通过 MkDocs strict 构建，中文跳转使用显式锚点。代理环境下运行工具时，需确保相关进程实际继承代理变量；
交互式 shell 中定义了代理，不代表非交互式下载命令已启用它。

## 准备 Minecraft { #minecraft-setup }

真实游戏仍使用仓库携带的 MineRL/Malmo 代码，需要 Java 8、可用的图形显示或 Xvfb/VirtualGL，以及 Minecraft 引擎。
此 alpha 已在 Linux / Python 3.12 / Java 8 / Xvfb 软件渲染上通过真实游戏与 VPT 1x 短程检查。
另已在 H800 / CUDA 13.0 / VirtualGL 3.1.5 上验证实际 GPU 渲染和 VPT foundation 1x / RL 2x 推理。其他硬件、操作系统和长时间采样仍需分别验证。

1. 从受信任来源取得 `engine.zip`。现有项目使用 [CraftJarvis/SimulatorEngine](https://huggingface.co/CraftJarvis/SimulatorEngine)。
2. 设置 `MINESTUDIO_DIR`，将引擎保存在持久目录。默认位置是系统临时目录下的 `MineStudio`。
3. 使用受信任的归档 SHA-256 显式安装。安装器拒绝不匹配的摘要、不安全的 ZIP 路径和覆盖已有引擎。

```bash
export MINESTUDIO_DIR=/path/to/minestudio-cache
minestudio engine install /path/to/engine.zip --sha256 EXPECTED_SHA256
minestudio doctor
```

引擎必须包含 `engine/build/libs/mcprec-6.13.jar`。SHA-256 用于校验归档，不代替可信的下载来源。
已有 v1 引擎也可通过相同 `MINESTUDIO_DIR` 发现；构造 `MinecraftEnv()` 不会交互询问或自动下载。
MineRL 在 reset info 中保存实际 JAR 摘要、安装 manifest（若存在）和环境配置，随轨迹落盘。
准备好显示服务后，显式运行真实引擎检查：

```bash
MINESTUDIO_GPU_RENDER=0 LIBGL_ALWAYS_SOFTWARE=1 MINESTUDIO_RUN_ENGINE=1 python -m pytest tests/e2e/test_minecraft.py::test_real_minerl_reset_step_close -q
```

上述 CPU 路径由启动脚本使用 `xvfb-run`，需要 Xvfb、xauth 与软件 OpenGL。
GPU 路径需另外配置 VirtualGL。支持范围与固定资源 revision 见[验证记录](../architecture/stage2-validation.md)。

## GPU 推理与渲染 { #gpu }

推理设备与渲染设备分别设置。安装适合驱动的 CUDA 版 Torch 后，通过
`VPTPolicy.from_pretrained(export, device="cuda:0")` 启用推理；`torch.cuda.is_available()` 只检查 CUDA 可用性，不证明游戏使用 GPU。

GPU 渲染还需要 VirtualGL 的 `vglrun`、`glxinfo`、X 显示服务，以及有权限访问的 NVIDIA DRM 节点。
设备选择使用 `cuda-bindings`（也兼容旧 `cuda-python`）；如当前 Torch 安装未提供它，可执行 `python -m pip install cuda-bindings`。

```bash
export CUDA_VISIBLE_DEVICES=0
export MINESTUDIO_POLICY_DEVICE=cuda:0
export MINESTUDIO_VPT_EXPORT=/path/to/verified-vpt-export
MINESTUDIO_GPU_RENDER=1 MINESTUDIO_RUN_ENGINE=1 xvfb-run -a python -m pytest tests/e2e -q
```

`MINESTUDIO_POLICY_DEVICE` 是 e2e 的配置入口，业务代码仍显式传 `device`。
`MINESTUDIO_GPU_RENDER=1` 按可见 GPU 顺序分配渲染设备，启动前检查真实 NVIDIA OpenGL；失败会报错。
默认仍是 CPU 渲染。VirtualGL 库必须能被 Java 加载；只有 `nvidia-smi` 显示设备或打印 DRM 路径不够。
本机实测、轨迹录像入口和库路径问题的处理见[GPU 验证记录](../architecture/gpu-validation.md)。

## 常见问题

| 现象 | 下一步 |
| --- | --- |
| `EngineNotFoundError` | 查看错误中的完整路径，检查 `MINESTUDIO_DIR` 和引擎标志文件 |
| `MissingDependencyError` | 安装错误提示对应的 extra |
| `EpisodeStateError` | 在首步或结束之后调用 `reset()`；已关闭的对象需要重新创建 |
| `BackendError` | 保留原异常链，检查游戏进程、Java、显示和通信日志 |
| `TaskMetricError` | 检查任务指标路径和后端遥测；不要用零掩盖缺失的父级数据 |
| FFmpeg 编码错误 | 查看视频旁的 `.ffmpeg.log`，检查 libx264 编码器和磁盘空间 |

[继续：第一次采样 →](quickstart.md)
