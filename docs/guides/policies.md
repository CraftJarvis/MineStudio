# 策略与模型

基础 `Policy` 是 Python 协议，导入它不加载 Torch、Ray 或 Lightning。状态由调用者持有，每条交互流独立。

```python
from minestudio.policies import Policy
```

## 实验性 VPT 适配器

安装 `.[policies]` 后可以访问 `VPTPolicy`。当前支持固定 VPT 动作编码和受限的 transformer 配置。
网络计算复用 v1 参数结构，通过新入口隔离在线训练依赖。

```python
from minestudio.policies.vpt import VPTConfig, VPTPolicy

policy = VPTPolicy(VPTConfig())  # 随机初始化；研究时需要加载合适的权重
state = policy.initial_state()
```

`act()` 返回解码后的原生动作。`act_with_stats()` 还提供实际采样的编码动作、behavior log probability 和反归一化后的 value。
`deterministic=True` 使用 argmax，此时行为分布是点质量，记录的 behavior log probability 为 `0`。

`act()` 使用推理模式；调用方需要保持 `eval()`。当前 alpha 尚未提供可用于 BC/PPO 的统一批处理接口。
Torch 继承关系不意味着这些训练能力已经完成。

## 本地模型导出

```python
policy.save_pretrained("runs/vpt-export")
restored = VPTPolicy.from_pretrained("runs/vpt-export", device="cpu")
```

目标目录必须不存在。导出包含 `manifest.json` 和 `model.safetensors`，绑定配置、codec、预处理版本和权重摘要。
加载时验证 manifest 和 SHA-256，并严格检查参数。该接口不自动访问 Hugging Face，不加载 pickle 文件。

## 迁移旧 VPT 权重 { #legacy-vpt }

`VPTConfig.from_legacy_kwargs(policy_kwargs, temperature=...)` 显式转换受支持的旧网络配置。
必须提供实际训练配置中的 `img_shape` 与 `timesteps`，不能用新配置的默认值猜测模型结构。
旧网络的缓存长度为 `attention_memory_size - timesteps`；v2 用 `sequence_length` 保留 timesteps，
同时保留 normalization、pre-LSTM layer norm、通道与 pointwise 配置。未知或不受支持的选项直接报错。

CLI 接收 JSON 配置，顶层格式为 `{"policy_kwargs": {...}, "temperature": 1.0}`，其中 policy_kwargs
必须来自对应权重的真实配置。Hub 配置中额外的 `action_space: null` 可直接接受，非默认动作空间会报错。
准备好这份文件与 tensor state dict 后执行：

```bash
minestudio convert-vpt --config /path/to/v1-config.json --weights /path/to/weights.safetensors \
  --output-dir artifacts/vpt --prefix mine_policy.
```

只在旧参数带有 `mine_policy.` 前缀时传入 `--prefix`，其他情况省略。
输入可以是 safetensors 或可由 `torch.load(weights_only=True)` 读取的纯张量 state dict，
也支持含 `state_dict` 的张量检查点。不加载序列化模型对象；输出拒绝覆盖已有目录。
导出 manifest 保存源权重摘要、配置摘要、原配置及移除的前缀。

Python 也可通过 `from_legacy_state_dict(weights, config=..., prefix=...)` 导入已读取的 state dict。
参数必须严格匹配，不能静默丢弃。VPT foundation 1x 发布权重已严格转换；CPU 上连续 136 帧的 v1/v2 网络输出、
确定性动作、反归一化价值与 recurrent state 逐项精确一致，覆盖 128 帧缓存长度。
CUDA 上已补充 foundation 1x 和挖钻石 RL 2x：分别以 400 步、2,000 步真实观测验证 v1/v2 网络、状态和随机动作精确一致。
加载到 GPU 时指定 `VPTPolicy.from_pretrained("runs/vpt-export", device="cuda:0")`，不要仅依据机器有 GPU 判断策略运行设备。
这些结论限定于[已验证的权重与运行版本](../architecture/gpu-validation.md)，其他权重、硬件和训练接口仍需单独验证。
GROOT、ROCKET、STEVE1 的新接口仍待迁移，当前入口保留在 `minestudio.models`。
