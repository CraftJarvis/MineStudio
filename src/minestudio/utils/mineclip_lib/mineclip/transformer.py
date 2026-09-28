"""
Transformer aggregator in temporal dimension.
https://github.com/lucidrains/x-transformers
pip install x_transformers==0.27.1
"""
from x_transformers.x_transformers import *
import torch

__all__ = ["TemporalTransformer", "make_temporal_transformer"]


class TemporalTransformer(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        embed_dim: int = None,
        depth: int,
        num_heads: int,
        max_seq_len: int,
        # ----- extra tricks, see x_transformers repo ----
        ff_glu=False,
        ff_swish=False,
        attn_one_kv_head=False,
        rel_pos_bias=False,
    ):
        """
        Reference arch:
            bert_base:
                embed_dim = 768
                depth = 12
                num_heads = 12
            bert_large:
                embed_dim = 1024
                depth = 24
                num_heads = 16

        Args:
            input_dim: continuous input feature dimension
            max_seq_len: max sequence length
            embed_dim: embedding dimension, if None, then it is the same as input_dim
                BUT will not add a projection layer from input -> first embedding
                if embed_dim is specified, a projection layer will be added even if
                input_dim == embed_dim
        """
        super().__init__()
        assert isinstance(max_seq_len, int)
        assert isinstance(input_dim, int)
        assert isinstance(depth, int)
        assert isinstance(num_heads, int)

        self.model = ContinuousTransformerWrapper(
            max_seq_len=max_seq_len,
            attn_layers=Encoder(
                dim=input_dim if embed_dim is None else embed_dim,
                depth=depth,
                heads=num_heads,
                ff_glu=ff_glu,
                ff_swish=ff_swish,
                attn_one_kv_head=attn_one_kv_head,
                rel_pos_bias=rel_pos_bias,
            ),
            # if embed_dim is None, do NOT add an input feature projection layer
            dim_in=None if embed_dim is None else input_dim,
            dim_out=None,
        )
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.num_heads = num_heads

    @property
    def output_dim(self):
        return self.input_dim if self.embed_dim is None else self.embed_dim

    def forward(self, x):
        B, L, F = x.size()
        x = self.model(x)
        x = x.mean(dim=1)
        assert x.shape == (B, self.output_dim)
        return x


def make_temporal_transformer(arch_spec: str, max_seq_len: int, input_dim=None):
    """
    Args:
        arch_spec: i<input_dim>[.e<embed_dim>].d<depth>.nh<num_heads>[.glu|glusw][.rel][.onekv]
            if embedding dimension is not specified, it is set to None and will not
            add a projection layer from input -> first embedding
        input_dim: if None, then must be specified in arch_spec as "i<input_dim>"
    """
    cfg = {}
    for spec in arch_spec.split("."):
        if spec.startswith("i"):
            cfg["input_dim"] = int(spec.removeprefix("i"))
        elif spec.startswith("e"):
            cfg["embed_dim"] = int(spec.removeprefix("e"))
        elif spec.startswith("d"):
            cfg["depth"] = int(spec.removeprefix("d"))
        elif spec.startswith("nh"):
            cfg["num_heads"] = int(spec.removeprefix("nh"))
        elif spec == "glu":
            cfg["ff_glu"] = True
        elif spec == "swglu" or spec == "glusw":
            cfg["ff_glu"] = True
            cfg["ff_swish"] = True
        elif spec == "rel":
            cfg["rel_pos_bias"] = True
        elif spec == "onekv":
            cfg["attn_one_kv_head"] = True
        else:
            raise ValueError(f'unknown spec part: "{spec}" in "{arch_spec}"')

    if "input_dim" not in cfg:
        if not input_dim:
            raise ValueError(
                f"input_dim must be specified in arch_spec: {arch_spec} or as input_dim arg"
            )
        cfg["input_dim"] = input_dim

    return TemporalTransformer(max_seq_len=max_seq_len, **cfg)
