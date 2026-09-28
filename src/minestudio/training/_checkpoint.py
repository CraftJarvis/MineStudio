"""Tensor-only RNG snapshots shared by local training checkpoints."""

import random
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), *state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def tensor_tree(tree: Any, device: str | torch.device) -> Any:
    """Detach and copy outside inference_mode before using caches in autograd."""
    if isinstance(tree, torch.Tensor):
        return tree.detach().to(device).clone()
    if isinstance(tree, list):
        return [tensor_tree(item, device) for item in tree]
    if isinstance(tree, tuple):
        return tuple(tensor_tree(item, device) for item in tree)
    if tree is None:
        return None
    raise TypeError(f"Unsupported recurrent cache node: {type(tree)}")
