"""Explicit tensor-only v1 export conversion. No implicit network access."""

import json
from pathlib import Path
from typing import Any

from minestudio.policies.vpt.policy import _digest as sha256_file


def convert_legacy_export(
    config_path: Path, weights_path: Path, output_dir: Path, *, prefix: str = ""
) -> Path:
    import torch
    from safetensors.torch import load_file

    from minestudio.policies.vpt import VPTConfig, VPTPolicy

    source: dict[str, Any] = json.loads(config_path.read_text())
    if (
        not isinstance(source, dict)
        or set(source) - {"policy_kwargs", "temperature", "action_space"}
        or "policy_kwargs" not in source
    ):
        raise ValueError(
            "Conversion config requires policy_kwargs, optional temperature/action_space"
        )
    if source.get("action_space") is not None:
        raise ValueError("Only the default VPT action_space (null) is supported")
    config = VPTConfig.from_legacy_kwargs(
        source["policy_kwargs"], temperature=source.get("temperature", 1.0)
    )
    if weights_path.suffix == ".safetensors":
        weights = load_file(str(weights_path), device="cpu")
    else:
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
    if not isinstance(weights, dict) or any(
        not isinstance(value, torch.Tensor) for value in weights.values()
    ):
        raise ValueError(
            "Expected a tensor-only state dict; arbitrary pickled models are unsupported"
        )
    policy = VPTPolicy.from_legacy_state_dict(weights, config=config, prefix=prefix)
    policy.save_pretrained(
        output_dir,
        provenance={
            "format": "v1-state-dict",
            "weights_sha256": sha256_file(weights_path),
            "config_sha256": sha256_file(config_path),
            "stripped_prefix": prefix,
            "source_config": source,
        },
    )
    return output_dir
