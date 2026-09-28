"""
Simple pooling aggregator in temporal dimension.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import make_temporal_transformer
from minestudio.utils.mineclip_lib.utils import build_mlp


class TemporalPooling(nn.Module):
    def __init__(
        self,
        *,
        pool_type,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers_before_pool: int,
        max_seq_len: int = None,
    ):
        """
        Args:
          pool_type:
            - "avg": average pooling
            - "attn.d8.nh8.rel...": see TemporalTransformer spec, always starts with
                "attn."; rest of the specs are separated by "."
            - None: do not pool at all, return [B, L, F] features
          layers_before_pool: number of MLP layers before pooling
        """
        super().__init__()
        assert pool_type in ["avg", None] or pool_type.startswith("attn.")

        self._pool_type = pool_type
        assert layers_before_pool >= 0
        self._layers_before_pool = layers_before_pool

        self.output_dim = output_dim

        self.residual_weight = None
        if layers_before_pool == 0:
            assert input_dim == output_dim, (
                "depth_before_pool is set to 0, therefore input_dim must be equal "
                "to output_dim because it is identity mapping. hidden_dim is ignored."
            )
            if pool_type == "catavgmax":
                assert (
                    output_dim == 2 * input_dim
                ), "output_dim must be 2 * input_dim for catavgmax"
            self.mlp_before_pool = nn.Identity()
        else:
            self.mlp_before_pool = build_mlp(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim // 2 if pool_type == "catavgmax" else output_dim,
                hidden_depth=layers_before_pool - 1,
                add_input_activation=False,
            )
            self.residual_weight = nn.Parameter(torch.tensor(4.0))

        if pool_type.startswith("attn."):
            assert input_dim == output_dim
            self.attn = make_temporal_transformer(
                pool_type.removeprefix("attn."),
                max_seq_len=max_seq_len,
                input_dim=input_dim,
            )
        else:
            self.attn = None

    def forward(self, x):
        B, L, F = x.size()
        if self.residual_weight is None:
            x = self.mlp_before_pool(x.view(B * L, F))
        else:
            res = torch.sigmoid(self.residual_weight)
            x = x.view(B * L, F)
            x = res * x + (1.0 - res) * self.mlp_before_pool(x)

        x = x.view(B, L, -1)
        if self._pool_type == "avg":
            x = x.mean(dim=1)
        elif self._pool_type in [None, "none"]:
            x = x
        elif "attn" in self._pool_type:
            # regular transformer already has positional embedding, pos_embed is None
            attn_out = self.attn(x)
            x = attn_out
        else:
            raise NotImplementedError
        if self._pool_type in ["none", None]:
            assert x.shape == (B, L, self.output_dim)
        else:
            assert x.shape == (B, self.output_dim)
        return x
