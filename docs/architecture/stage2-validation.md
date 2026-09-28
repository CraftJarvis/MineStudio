# 阶段二验证记录

日期：2026-09-28。版本：`2.0.0a2`。本记录包含本机实际执行结果；GPU 和较长轨迹见[补充验证](gpu-validation.md)。CI 与其他平台尚未验证。

## 网络问题与运行环境

先前下载超时的原因是当前非交互式工具进程没有 HTTP/HTTPS/ALL_PROXY，尽管 shell 配置中已有可用代理。
直连 PyPI 和 Hugging Face 在 TCP 连接阶段超时；显式使用现有代理后均返回 HTTP 200，约 0.3–0.7 秒。
代理仅用于本次下载命令，没有写入仓库或修改用户的全局 shell 配置。

`.venv-stage2` 现在包含完整的已测试运行依赖：

| 组件 | 本机版本/路径 |
| --- | --- |
| Python / NumPy | 3.12.3 / 1.26.4 |
| Torch | 2.14.0+cpu |
| Gymnasium / gym / gym3 | 1.3.0 / 0.26.2 / 0.3.3 |
| OpenCV / safetensors | 4.11.0.86 / 0.8.0 |
| Java / 渲染 | Java 8，Xvfb + 软件 OpenGL |
| MkDocs / Material | 1.6.1 / 9.7.7 |
| 检查工具 | pytest 9.1.1、Ruff 0.16.9、Pyright 1.1.414 |

后续已新建 `.venv-stage2-gpu`（Torch 2.14.0+cu130），在单张 H800 上验证 CUDA 推理与实际 NVIDIA OpenGL 渲染。
原先的 CPU 检查保留；设备证据、浮点误差和行为分析见[GPU 与轨迹行为验证](gpu-validation.md)。

## 固定资源身份 { #fixed-resources }

| 资源 | 仓库与 revision | 文件 SHA-256 |
| --- | --- | --- |
| 引擎 engine.zip | `CraftJarvis/SimulatorEngine`，`48d4809cfddc7e2b85295e8c39b3c5e8c6d46ae7` | `293fac6ac72245b3365dce0e8bfbb6396fb94df29b23b6538f3bd7e2eec13ec6` |
| VPT 1x model.safetensors | `CraftJarvis/MineStudio_VPT.foundation_model_1x`，`17a5f43b30c4f734489902fdc6a55bf47781be3a` | `475fbd0df655ad77c3e3f602d157f4273032bff8e6e82c3863a992f5b03753f9` |
| VPT 配置 config.json | 与上述权重同 revision | `d088a1f68ca44cac73d0efe1af7b7df4ade5994b39360da7fe74cfb6b282cbd2` |

大文件按 Hub LFS 提供的 SHA-256 校验，再由显式安装/转换入口处理。
下载、引擎、导出与日志位于项目内忽略的 `runs/stage2-runtime`；不进入发行包。

## 已执行检查

| 检查 | 结果 |
| --- | --- |
| Ruff / Pyright strict | 当前配置范围通过；补装 OpenCV 类型信息后修正了图像返回类型 |
| unit + integration | **CPU 74 / CUDA 环境 75 passed，均无跳过**，包含 VPT 网络/导出与 MineRL 导入 |
| 真实引擎 | reset/step/close e2e 通过，约 53 秒，末步预算截断 |
| 真实 VPT 链路 | 已转换发布权重 → 游戏任务 → 3 步 → 轨迹读回/视频 e2e 通过，约 54 秒 |
| 发布权重数值对照 | CPU 连续 136 帧，跨过 128 帧缓存长度；网络输出、确定性动作、反归一化价值和 recurrent state 精确一致，rtol=atol=0 |
| GPU 与行为 | foundation 1x / RL 2x 源张量精确加载，2,400 步真实观测 v1/v2 对照；四段轨迹共 3,200 步，含 2,400 步实际 GPU 渲染 |
| FFmpeg/ffprobe | 首末帧、帧数、奇数尺寸补齐与编码器关闭通过 |
| 故障语义 | 重试保留 attempts；中断恢复有效前缀；报告错误不覆盖原始异常；逐一关闭所有 writer |
| MkDocs | strict 构建通过，锚点检查设为 warn 并修复中文跳转 |
| 浏览器 | Chromium 1440×1000 / 390×844：明暗切换、移动导航、搜索、实际剪贴板复制、锚点通过；无页面 JS 异常；已查看截图并改善窄屏图表阅读 |
| 打包 | a2 wheel/sdist；仓库外最小安装、资源与轨迹读回通过；sdist 包含文档、示例和测试 |
| 原路径保留 | 842 个原包路径全部位于 src；wheel 排除历史 MP4、Git/编辑器与陈旧 FUSE 辅助文件 |

数值对照直接执行原始 v1 网络类定义，复用保留的 head 组件；输入为 RNG seed 13 生成的 128×128 RGB 序列。
不代表其他模型、设备、训练路径或任意输入上的统一误差保证。
此处的初始真实游戏 e2e 使用采木任务，预算为 3 步，结果为 `task_time_limit`、success=false；未声称任务采集成功。

旧 gym 提示弃用信息，MineRL 日志线程使用的旧接口也有 DeprecationWarning；测试通过不表示兼容区域已全部现代化。
MkDocs 的静态解析还提示旧模块 docstring 转义警告，当前按 INFO 记录，不影响 strict 构建。

## 复现命令

```bash
source .venv-stage2/bin/activate
ruff check .
ruff format --check .
pyright
python -m pytest -q
mkdocs build --strict
```

本机资源就绪后，执行真实 e2e：

```bash
export MINESTUDIO_DIR="$PWD/runs/stage2-runtime"
export MINESTUDIO_VPT_EXPORT="$PWD/runs/stage2-runtime/vpt-1x"
MINESTUDIO_GPU_RENDER=0 LIBGL_ALWAYS_SOFTWARE=1 MINESTUDIO_RUN_ENGINE=1 python -m pytest tests/e2e -q
```

数值报告为 `runs/stage2-runtime/vpt-parity.json`。
浏览器检查与截图位于 `runs/stage2-runtime/docs-qa/`；相关日志保存在 `runs/stage2-runtime/`。
真实游戏评测报告为
`runs/stage2-runtime/e2e-vpt/test_real_vpt_task_artifacts0/99e7412baf594e589589ea9e1df07328/evaluation.json`。
同一运行目录保存轨迹，视频位于其 `videos` 子目录。无引擎的合成示例仍见[快速入门](../getting-started/quickstart.md)。

## 尚未验证

其他 GPU/驱动、其余 VPT 变体、其他操作系统/Python 运行时、长期稳定性、固定种子下完整游戏轨迹的确定性，
以及较长预算下的任务成功率仍需验证。远端 CI 未运行。训练、历史数据迁移和分布式执行属于后续阶段。
`2.0.0a2` 没有发布到 PyPI，文档站点没有部署。
