'''
Date: 2024-12-25 23:39:41
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-12-30 14:27:30
FilePath: /MineStudio/minestudio/utils/__init__.py
'''
from .temp import get_mine_studio_dir


def get_compute_device() -> "torch.device":
    """
    Return the optimal `torch.device` for this machine.

    Preference order:
    1. CUDA GPU
    2. Apple-Silicon (MPS)
    3. CPU
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Canonical MPS check recommended by PyTorch docs
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def __getattr__(name):
    if name in {"Register", "Registers"}:
        from . import register
        return getattr(register, name)
    raise AttributeError(name)
