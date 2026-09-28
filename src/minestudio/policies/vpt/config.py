"""Explicit configuration for the supported VPT transformer architecture."""

import math
from dataclasses import dataclass, field

from minestudio.core import ImageSize


@dataclass(frozen=True)
class VPTConfig:
    image_size: ImageSize = field(default_factory=lambda: ImageSize(128, 128))
    hidden_size: int = 512
    attention_heads: int = 8
    attention_memory_size: int = 128
    num_layers: int = 4
    channel_multiplier: int = 1
    temperature: float = 1.0
    sequence_length: int = 1
    channels: tuple[int, ...] = (16, 32, 32)
    group_norm_groups: int | None = None
    batch_norm: bool = False
    post_pool_groups: int | None = None
    first_conv_norm: bool = False
    pre_recurrent_layer_norm: bool = True
    use_pointwise_layer: bool = True
    pointwise_ratio: int = 4
    pointwise_activation: bool = False
    recurrent_residual: bool = True
    scale_input_image: bool = True

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "attention_heads",
            "attention_memory_size",
            "num_layers",
            "channel_multiplier",
            "sequence_length",
            "pointwise_ratio",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.attention_memory_size <= self.sequence_length:
            raise ValueError("attention_memory_size must exceed sequence_length")
        if (
            not isinstance(self.channels, tuple)
            or not self.channels
            or any(type(c) is not int or c <= 0 for c in self.channels)
        ):
            raise ValueError("channels must be a tuple of positive integers")
        for name in ("group_norm_groups", "post_pool_groups"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        for name in (
            "batch_norm",
            "first_conv_norm",
            "pre_recurrent_layer_norm",
            "use_pointwise_layer",
            "pointwise_activation",
            "recurrent_residual",
            "scale_input_image",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

    @classmethod
    def from_legacy_kwargs(
        cls, policy_kwargs: dict[str, object], *, temperature: float = 1.0
    ) -> "VPTConfig":
        """Translate recognized v1 transformer options without dropping unknown keys.

        In the retained network, cache length equals memory_size - timesteps.
        Preserving timesteps is necessary even when inference processes one frame.
        """
        from typing import Any, cast

        data: dict[str, Any] = dict(policy_kwargs)
        mapping = {
            "hidsize": "hidden_size",
            "attention_heads": "attention_heads",
            "attention_memory_size": "attention_memory_size",
            "n_recurrence_layers": "num_layers",
            "impala_width": "channel_multiplier",
            "timesteps": "sequence_length",
            "first_conv_norm": "first_conv_norm",
            "use_pre_lstm_ln": "pre_recurrent_layer_norm",
            "use_pointwise_layer": "use_pointwise_layer",
            "pointwise_ratio": "pointwise_ratio",
            "pointwise_use_activation": "pointwise_activation",
            "recurrence_is_residual": "recurrent_residual",
            "scale_input_img": "scale_input_image",
        }
        fixed = {
            "recurrence_type": "transformer",
            "single_output": False,
            "attention_mask_style": "clipped_causal",
            "img_statistics": None,
            "diff_mlp_embedding": False,
        }
        # These are explicitly unused by the original MinecraftPolicy constructor.
        unused = {"obs_processing_width", "only_img_input", "input_shape", "active_reward_monitors"}
        allowed = (
            set(mapping)
            | set(fixed)
            | unused
            | {"img_shape", "impala_chans", "init_norm_kwargs", "impala_kwargs"}
        )
        if set(data) - allowed:
            raise ValueError(f"Unsupported legacy VPT options: {sorted(set(data) - allowed)}")
        for name, expected in fixed.items():
            if name in data and data[name] != expected:
                raise ValueError(f"Unsupported legacy VPT {name}={data[name]!r}")
        translated: dict[str, Any] = {
            target: data[source] for source, target in mapping.items() if source in data
        }
        if "timesteps" not in data or data["timesteps"] is None:
            raise ValueError("Legacy timesteps must be explicit to preserve the attention cache")
        shape = data.get("img_shape")
        if not isinstance(shape, (list, tuple)):
            raise ValueError("Legacy img_shape must explicitly specify [height, width, 3]")
        dimensions = cast(list[Any] | tuple[Any, ...], shape)
        if len(dimensions) != 3 or dimensions[2] != 3:
            raise ValueError("Legacy img_shape must explicitly specify [height, width, 3]")
        translated["image_size"] = ImageSize(dimensions[0], dimensions[1])
        if "impala_chans" in data:
            translated["channels"] = tuple(data["impala_chans"])
        for group, keys in (
            (
                "init_norm_kwargs",
                {"group_norm_groups": "group_norm_groups", "batch_norm": "batch_norm"},
            ),
            ("impala_kwargs", {"post_pool_groups": "post_pool_groups"}),
        ):
            options = data.get(group, {})
            if not isinstance(options, dict) or set(cast(dict[str, Any], options)) - set(keys):
                raise ValueError(f"Unsupported legacy {group}")
            for old, new in keys.items():
                if old in options:
                    translated[new] = options[old]
        return cls(**translated, temperature=temperature)

    def network_kwargs(self) -> dict[str, object]:
        """Exact kwargs for the retained network; normalization and cache are explicit."""
        return {
            "hidsize": self.hidden_size,
            "img_shape": [self.image_size.height, self.image_size.width, 3],
            "impala_width": self.channel_multiplier,
            "impala_chans": self.channels,
            "attention_heads": self.attention_heads,
            "attention_memory_size": self.attention_memory_size,
            "n_recurrence_layers": self.num_layers,
            "timesteps": self.sequence_length,
            "init_norm_kwargs": {
                "group_norm_groups": self.group_norm_groups,
                "batch_norm": self.batch_norm,
            },
            "impala_kwargs": {"post_pool_groups": self.post_pool_groups},
            "first_conv_norm": self.first_conv_norm,
            "use_pre_lstm_ln": self.pre_recurrent_layer_norm,
            "use_pointwise_layer": self.use_pointwise_layer,
            "pointwise_ratio": self.pointwise_ratio,
            "pointwise_use_activation": self.pointwise_activation,
            "recurrence_is_residual": self.recurrent_residual,
            "scale_input_img": self.scale_input_image,
        }
