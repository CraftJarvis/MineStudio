"""
Adapted from OpenAI CLIP implementation: https://github.com/openai/CLIP
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from .pos_embed import interpolate_resize_pos_embed
from .tokenization import tokenize_batch
import minestudio.utils.mineclip_lib.utils as U


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self._resolution = resolution
        self._patch_size = patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        self.pos_embed = nn.Parameter(
            scale * torch.randn((resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

    def resize_pos_embed(self, new_resolution):
        """
        NOTE: call this method AFTER you load pretrained weights!
        """
        if isinstance(new_resolution, int):
            new_resolution = (new_resolution, new_resolution)
        else:
            assert len(new_resolution) == 2
        for r in new_resolution:
            assert (
                r % self._patch_size == 0
            ), f"{new_resolution} is not divisible by {self._patch_size}"

        with torch.no_grad():
            old_embed = self.pos_embed.data.detach()
            cls_embed, old_embed = old_embed[:1], old_embed[1:]
            new_embed = interpolate_resize_pos_embed(
                old_embed,
                self._resolution // self._patch_size,
                [r // self._patch_size for r in new_resolution],
            )
            self.pos_embed = nn.Parameter(torch.cat([cls_embed, new_embed], dim=0))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.repeat((B, 1, 1)), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.projection is not None:
            x = x @ self.projection

        return x


class GPT(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        layers: int,
        width: int,
        heads: int,
        is_discrete_text: bool = True,
    ):
        """
        Args:
            is_discrete_text: False to use regular discrete tokens
              True for video sequence of image tokens, and `vocab_size` will be
              interpreted as the dim of each image feature.
        """
        super().__init__()
        self.context_length = context_length
        self._width = width
        self._layers = layers
        self.vocab_size = vocab_size

        self._is_discrete_text = is_discrete_text
        if is_discrete_text:
            self.token_embedding = nn.Embedding(vocab_size, width)
        else:
            self.token_embedding = nn.Linear(vocab_size, width, bias=False)
        self.pos_embed = nn.Parameter(torch.empty(self.context_length, width))
        self.blocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width, heads, attn_mask=self.build_attention_mask()
                )
                for _ in range(layers)
            ]
        )

        self.ln_final = nn.LayerNorm(width)
        self.projection = nn.Parameter(torch.empty(width, embed_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        if self._is_discrete_text:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

        proj_std = (self._width**-0.5) * ((2 * self._layers) ** -0.5)
        attn_std = self._width**-0.5
        fc_std = (2 * self._width) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.projection is not None:
            nn.init.normal_(self.projection, std=self._width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        assert (
            x.size(1) <= self.context_length
        ), f"{x.size(1)} exceeds context length {self.context_length}"
        x = x + self.pos_embed  # x = x + self.pos_embed[: x.size(1)]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self._is_discrete_text:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.projection
        else:
            # last token will be the GPT summary
            x = x[:, -1] @ self.projection
        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        text_layers: int,
        text_width: int,
        text_heads: int,
    ):
        super().__init__()

        vision_heads = vision_width // 64
        self.vision_model = VisionTransformer(
            resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
        )
        self.text_model = GPT(
            embed_dim=embed_dim,
            context_length=context_length,
            vocab_size=vocab_size,
            layers=text_layers,
            width=text_width,
            heads=text_heads,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        return self.vision_model(image)

    def tokenize_text(self, text: str | list[str]):
        if isinstance(text, list):
            assert len(text) > 0
            assert isinstance(text[0], str), "only supports str or list[str]"
        return tokenize_batch(text, max_length=77, language_model="clip")

    def encode_text(self, text):
        if isinstance(text, str) or isinstance(text, list):
            tokens = self.tokenize_text(text)
            return self.encode_text(tokens.to(device=U.get_device(self.text_model)))
        elif text.dtype == torch.long:
            return self.text_model(text)
        else:
            return text

    def forward(self, image, text):
        if image.ndim == 2:
            image_features = image
        else:
            image_features = self.encode_image(image)
        if text.dtype == torch.long:
            text_features = self.encode_text(text)
        else:
            text_features = text

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    @torch.no_grad()
    def clamp_logit_scale(self, value=100):
        """
        Follow OpenAI CLIP paper's trick to prevent training instability (sec 2.5)
        """
        self.logit_scale.data.clamp_(-np.log(value), np.log(value))
