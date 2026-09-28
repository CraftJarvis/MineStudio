from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import tree

from .file_utils import *


def torch_normalize(tensor: torch.Tensor, mean, std, inplace=False):
    """
    Adapted from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#normalize

    Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("tensor should be a torch tensor. Got {}.".format(type(tensor)))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
        )
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor


def torch_load(*fpath: str, map_location="cpu") -> dict:
    """
    Default maps to "cpu"
    """
    fpath = str(f_join(fpath))
    try:
        return torch.load(fpath, map_location=map_location)
    except RuntimeError as e:
        raise RuntimeError(f"{e}\n\n --- Error loading {fpath}")


def load_state_dict(objects, states, strip_prefix=None, strict=False):
    """
    Args:
        strict: objects and states must match exactly
        strip_prefix: only match the keys that have the prefix, and strip it
    """

    def _load(paths, obj):
        if not _implements_method(obj, "load_state_dict"):
            raise ValueError(
                f"Object {type(obj)} does not support load_state_dict() method"
            )
        try:
            state = _tree_value_at_path(states, paths)
        except ValueError:  # paths do not exist in `states` structure
            if strict:
                raise
            else:
                return
        if strip_prefix:
            assert isinstance(strip_prefix, str)
            state = {
                k[len(strip_prefix) :]: v
                for k, v in state.items()
                if k.startswith(strip_prefix)
            }
        if isinstance(obj, nn.Module):
            return obj.load_state_dict(state, strict=strict)
        else:
            return obj.load_state_dict(state)

    return tree.map_structure_with_path(_load, objects)


def _implements_method(object, method: str):
    """
    Returns:
        True if object implements a method
    """
    return hasattr(object, method) and callable(getattr(object, method))


def _tree_value_at_path(obj, paths: tuple):
    try:
        for p in paths:
            obj = obj[p]
        return obj
    except Exception as e:
        raise ValueError(f"{e}\n\n-- Incorrect nested path {paths} for object: {obj}.")


def get_activation(activation: str | Callable | None) -> Callable:
    if not activation:
        return nn.Identity
    elif callable(activation):
        return activation
    ACT_LAYER = {
        "tanh": nn.Tanh,
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
        "swish": lambda: nn.SiLU(inplace=True),  # SiLU is alias for Swish
        "sigmoid": nn.Sigmoid,
        "elu": lambda: nn.ELU(inplace=True),
        "gelu": nn.GELU,
    }
    activation = activation.lower()
    assert activation in ACT_LAYER, f"Supported activations: {ACT_LAYER.keys()}"
    return ACT_LAYER[activation]


def get_initializer(method: str | Callable, activation: str) -> Callable:
    if isinstance(method, str):
        assert hasattr(
            nn.init, f"{method}_"
        ), f"Initializer nn.init.{method}_ does not exist"
        if method == "orthogonal":
            try:
                gain = nn.init.calculate_gain(activation)
            except ValueError:
                gain = 1.0
            return lambda x: nn.init.orthogonal_(x, gain=gain)
        else:
            return getattr(nn.init, f"{method}_")
    else:
        assert callable(method)
        return method


def set_requires_grad(model, requires_grad):
    if torch.is_tensor(model):
        model.requires_grad = requires_grad
    else:
        for param in model.parameters():
            param.requires_grad = requires_grad


def get_device(x, strict: bool = False) -> int:
    """
    Args:
        x: can be any arbitrary nested structure of np array and torch tensor
        strict: True to check all batch sizes are the same
    """
    xs = tree.flatten(x)

    def _get_device(x):
        if torch.is_tensor(x):
            return x.device
        elif isinstance(x, nn.Module):
            return get_module_device(x)
        else:
            return None

    if strict:
        devices = [_get_device(x) for x in xs]
        assert all(
            b == devices[0] for b in devices
        ), f"devices must all be the same in nested structure: {devices}"
        return devices[0]
    else:
        return _get_device(xs[0])


def get_module_device(model):
    """
    Returns:
        first model parameter's device
    """
    return next(model.parameters()).device
