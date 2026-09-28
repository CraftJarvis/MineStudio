"""
Base API for importing pretrained video models
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional

import minestudio.utils.mineclip_lib.utils as U

__all__ = ["VideoRewardBase"]

# calculated from 21K video clips, which contains 2.8M frames
MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)


class VideoRewardBase(nn.Module):
    def __init__(
        self,
        *,
        image_encoder: nn.Module,
        temporal_encoder: nn.Module,
        reward_head: nn.Module,
    ):
        """
        Args:
          image_encoder: [B, C, H, W] -> [B, F]
          temporal_encoder: [B, L, F] -> [B, F]
          reward_head: [B, F] -> [B, D] softmax over D classes/dims
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.temporal_encoder = temporal_encoder
        self.reward_head = reward_head

    def forward_image_features(self, frames):
        """
        [..., C, H, W] -> [..., F], independent encoding of each frame image
        """
        assert frames.ndim >= 4
        leading_dims = frames.size()[:-3]
        C, H, W = frames.size()[-3:]
        frames = frames.view(-1, C, H, W)
        frames = U.basic_image_tensor_preprocess(
            frames, mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD
        )
        features = self.image_encoder(frames)
        return features.view(*leading_dims, features.size(-1))

    def forward_video_features(self, image_features):
        """
        [B, L, F] -> [B, F]
        """
        B, L, F = image_features.size()
        video_feats = self.temporal_encoder(image_features)
        assert video_feats.shape[0] == B
        return video_feats

    def forward_reward_head(self, video_features, text_tokens=None, softmax=False):
        """
        [B, F] -> [B, D]
        """
        B, F = video_features.size()
        if text_tokens is not None:
            rewards = self.reward_head(video_features, text_tokens)
        else:
            rewards = self.reward_head(video_features)
        if torch.is_tensor(rewards):
            assert rewards.shape[0] == B
            if softmax:
                rewards = torch.nn.functional.softmax(rewards, dim=1)
        return rewards

    def forward(self, videos, text_tokens=None, is_video_features=False):
        """
        Args:
            videos: [B, F] if is_video_features else [B, L, C, H, W]
            is_video_features: pass in [B, F] of already-computed video features
            text_tokens: [B, L, D]
        """
        if is_video_features:
            assert videos.ndim == 2
            return self.forward_reward_head(videos, text_tokens=text_tokens)
        else:
            assert videos.ndim == 5, "video must be 5D (raw pixels)"
            return self.forward_reward_head(
                self.forward_video_features(self.forward_image_features(videos)),
                text_tokens=text_tokens,
            )

    def load_ckpt(self, ckpt_or_path, strip_prefix="model.", strict=False):
        if isinstance(ckpt_or_path, dict):
            ckpt = ckpt_or_path
        else:
            ckpt_path = U.f_expand(ckpt_or_path)
            assert U.f_exists(ckpt_path), f"ckpt not found: {ckpt_path}"
            ckpt = U.torch_load(ckpt_path)
        # `ret` might contain key matching info if strict=False
        ret = U.load_state_dict(
            self, ckpt["state_dict"], strip_prefix=strip_prefix, strict=strict
        )
        return ret
