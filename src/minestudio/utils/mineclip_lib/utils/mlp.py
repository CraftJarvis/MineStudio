from __future__ import annotations
import torch.nn as nn
from typing import Callable, Literal
from .torch_utils import get_activation, get_initializer

__all__ = ["build_mlp", "MLP"]


def build_mlp(
    input_dim,
    *,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: int = None,
    num_layers: int = None,
    activation: str | Callable = "relu",
    weight_init: str | Callable = "orthogonal",
    bias_init="zeros",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    add_input_activation: bool | str | Callable = False,
    add_input_norm: bool = False,
    add_output_activation: bool | str | Callable = False,
    add_output_norm: bool = False,
) -> nn.Sequential:
    """
    In other popular RL implementations, tanh is typically used with orthogonal
    initialization, which may perform better than ReLU.

    Args:
        norm_type: None, "batchnorm", "layernorm", applied to intermediate layers
        add_input_activation: whether to add a nonlinearity to the input _before_
            the MLP computation. This is useful for processing a feature from a preceding
            image encoder, for example. Image encoder typically has a linear layer
            at the end, and we don't want the MLP to immediately stack another linear
            layer on the input features.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_input_norm: see `add_input_activation`, whether to add a normalization layer
            to the input _before_ the MLP computation.
            values: True to add the `norm_type` to the input
        add_output_activation: whether to add a nonlinearity to the output _after_ the
            MLP computation.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_output_norm: see `add_output_activation`, whether to add a normalization layer
            _after_ the MLP computation.
            values: True to add the `norm_type` to the input
    """
    assert (hidden_depth is None) != (num_layers is None), (
        "Either hidden_depth or num_layers must be specified, but not both. "
        "num_layers is defined as hidden_depth+1"
    )
    if hidden_depth is not None:
        assert hidden_depth >= 0
    if num_layers is not None:
        assert num_layers >= 1
    act_layer = get_activation(activation)

    weight_init = get_initializer(weight_init, activation)
    bias_init = get_initializer(bias_init, activation)

    if norm_type is not None:
        norm_type = norm_type.lower()

    if not norm_type:
        norm_type = nn.Identity
    elif norm_type == "batchnorm":
        norm_type = nn.BatchNorm1d
    elif norm_type == "layernorm":
        norm_type = nn.LayerNorm
    else:
        raise ValueError(f"Unsupported norm layer: {norm_type}")

    hidden_depth = num_layers - 1 if hidden_depth is None else hidden_depth
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), norm_type(hidden_dim), act_layer()]
        for i in range(hidden_depth - 1):
            mods += [
                nn.Linear(hidden_dim, hidden_dim),
                norm_type(hidden_dim),
                act_layer(),
            ]
        mods.append(nn.Linear(hidden_dim, output_dim))

    if add_input_norm:
        mods = [norm_type(input_dim)] + mods
    if add_input_activation:
        if add_input_activation is not True:
            act_layer = get_activation(add_input_activation)
        mods = [act_layer()] + mods
    if add_output_norm:
        mods.append(norm_type(output_dim))
    if add_output_activation:
        if add_output_activation is not True:
            act_layer = get_activation(add_output_activation)
        mods.append(act_layer())

    for mod in mods:
        if isinstance(mod, nn.Linear):
            weight_init(mod.weight)
            bias_init(mod.bias)

    return nn.Sequential(*mods)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        *,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int = None,
        num_layers: int = None,
        activation: str | Callable = "relu",
        weight_init: str | Callable = "orthogonal",
        bias_init="zeros",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        add_input_activation: bool | str | Callable = False,
        add_input_norm: bool = False,
        add_output_activation: bool | str | Callable = False,
        add_output_norm: bool = False,
    ):
        super().__init__()
        # delegate to build_mlp by keywords
        self.layers = build_mlp(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
            num_layers=num_layers,
            activation=activation,
            weight_init=weight_init,
            bias_init=bias_init,
            norm_type=norm_type,
            add_input_activation=add_input_activation,
            add_input_norm=add_input_norm,
            add_output_activation=add_output_activation,
            add_output_norm=add_output_norm,
        )
        # add attributes to the class
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_depth = hidden_depth
        self.activation = activation
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.norm_type = norm_type
        if add_input_activation is True:
            self.input_activation = activation
        else:
            self.input_activation = add_input_activation
        if add_input_norm is True:
            self.input_norm_type = norm_type
        else:
            self.input_norm_type = None
        # do the same for output activation and norm
        if add_output_activation is True:
            self.output_activation = activation
        else:
            self.output_activation = add_output_activation
        if add_output_norm is True:
            self.output_norm_type = norm_type
        else:
            self.output_norm_type = None

    def forward(self, x):
        return self.layers(x)
