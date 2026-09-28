import torch
import numpy as np
from einops import rearrange


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, seq_len: int, cls_token=False):
    """
    Returns:
        [seq_len, embed_dim] or [1+seq_len, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(seq_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return torch.tensor(pos_embed, dtype=torch.float)


def interpolate_resize_pos_embed(pos_embed, old_size, new_size):
    """
    NOTE: remove cls token from pos_embed first before passing it here

    Args:
        pos_embed: [seq_len, embed_dim]
        old_size: [h, w], seq_len of pos_embed must be equal to h * w
        new_size: [new_h, new_w]
    """
    old_hw, D = pos_embed.size()
    if isinstance(old_size, int):
        old_size = (old_size, old_size)
    if isinstance(new_size, int):
        new_size = (new_size, new_size)
    assert len(old_size) == 2
    assert len(new_size) == 2
    old_h, old_w = old_size
    assert old_h * old_w == old_hw
    pos_embed = rearrange(pos_embed, "(H W) D -> 1 D H W", H=old_h)
    new_embed = torch.nn.functional.interpolate(
        pos_embed, size=new_size, mode="bicubic", align_corners=False
    )
    new_embed = rearrange(new_embed, "1 D H W -> (H W) D")
    assert new_embed.size() == (new_size[0] * new_size[1], D)
    return new_embed
