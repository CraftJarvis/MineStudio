# 环境与动作

## 公共数据格式

`Observation.image` 是 `uint8` RGB 数组，形状为 `(height, width, 3)`。使用 `ImageSize(height=224, width=224)` 指定尺寸。

`MinecraftAction` 包含完整的具名 `buttons` 字典和 `float32`、形状 `(2,)` 的 `camera` 数组。
相机顺序是 `(pitch_delta, yaw_delta)`，单位为度。按钮接受整数 `0`/`1`；`noop_action()` 每次创建新对象。

```python
from minestudio.actions import noop_action

action = noop_action()
action["buttons"]["jump"] = 1
action["camera"][:] = [0, 5]
```

环境接收原生动作。VPT 的 `buttons=123` 等离散编码不能直接传给 `step()`；应由 `VPTActionCodec` 解码。
codec 会量化与裁剪相机，消解冲突按键，因此不是任意输入的无损双向转换。

## 生命周期

构造 → `reset(seed=...)` → `step(action)` → 正常终止或预算截断 → `reset()` 或 `close()`。

- 首次 `reset()` 创建后端。构造阶段只验证配置。
- `step()` 返回 Gymnasium 五元组，保留最后观测，禁止自动重置。
- `close()` 可重复调用。已关闭的对象不允许重启。
- 引擎通信错误抛出 `BackendError`；不将失败包装成一个随机观测的终止步。
- 图像和 info 在边界复制，避免后端复用 buffer 改写过去的数据。

`render_mode="rgb_array"` 返回最近的策略分辨率图像副本；`render_size` 指定游戏后端的原始渲染尺寸。
`MinecraftEnv(task=...)` 应用声明式任务，详见[任务指南](tasks.md)。
当前 alpha 不提供人类交互窗口，通用环境 wrappers 仍待迁移。

## 接入后端

`MinecraftEnv(..., backend_factory=factory)` 是显式扩展点。工厂接收 `EnvConfig`，每次返回新的 `EnvBackend`。
后端实现 `reset`、`step`、`close`，返回与 `image_size` 匹配的图像；环境负责其生命周期。
若任务包含初始化命令，还需要实现 `execute_command(command)` 并返回执行后的观测和 info。
第三方后端必须将通信故障表示为异常，不能返回虚构转移。

环境种子通过每个实例的 RNG 生成并在 reset info 中记录。底层 Minecraft/Malmo 是否完全确定性仍需游戏验证。
