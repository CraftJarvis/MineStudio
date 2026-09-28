from __future__ import annotations

import re

import torch
import torch.nn as nn

from .base import VideoRewardBase
from .clip import CLIP
from .pooling import TemporalPooling
from .head import CLIPScoreHead


class MineCLIP(VideoRewardBase):
    def __init__(
        self,
        arch: str,
        *,
        resolution: tuple[int, int],
        pool_type: str,
        image_feature_dim: int,
        mlp_adapter_spec: str,
        hidden_dim: int,
    ):
        """
        Args:
          mlp_adapter_spec: v3-1.t2 means on the vision branch, 3 MLP layers of image
            adapter (before video pooling), 1 layer of video adapter (after pooling).
            On text branch, 2 layers of text adapter
        """
        self.arch = arch
        VIDEO_SEQ_LEN = 32
        assert arch.startswith("vit_base_p16")
        assert image_feature_dim == 512
        clip_config = {
            "context_length": 77,
            "embed_dim": 512,
            "image_resolution": 224,
            "text_heads": 8,
            "text_layers": 12,
            "text_width": 512,
            "vision_layers": 12,
            "vision_patch_size": 16,
            "vision_width": 768,
            "vocab_size": 49408,
        }
        model = CLIP(**clip_config)
        model.vision_model.resize_pos_embed(resolution)

        # regex match v3-1.t2
        m = re.match(
            r"v(?P<image_adapter>\d+)"
            r"-(?P<video_adapter>\d+)"
            r"\.t(?P<text_adapter>\d+)",
            mlp_adapter_spec,
        )
        image_adapter_layers, video_adapter_layers, text_adapter_layers = (
            int(m.group("image_adapter")),
            int(m.group("video_adapter")),
            int(m.group("text_adapter")),
        )

        assert image_feature_dim == hidden_dim

        temporal_encoder = TemporalPooling(
            pool_type=pool_type,
            input_dim=image_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            layers_before_pool=image_adapter_layers,
            max_seq_len=VIDEO_SEQ_LEN,
        )
        if not isinstance(temporal_encoder.mlp_before_pool, nn.Identity):
            for module in temporal_encoder.mlp_before_pool:
                # initialize linear layers as identity matrices
                if isinstance(module, nn.Linear):
                    module.weight.data.copy_(torch.eye(module.weight.shape[0]))
                    module.bias.data.zero_()
        reward_head = CLIPScoreHead(
            model,
            video_adapter_layers=video_adapter_layers,
            text_adapter_layers=text_adapter_layers,
            feature_dim=image_feature_dim,
        )

        super().__init__(
            image_encoder=model.vision_model,
            temporal_encoder=temporal_encoder,
            reward_head=reward_head,
        )
        self.clip_model = model

    def encode_text(self, text_tokens):
        return self.clip_model.encode_text(text_tokens)

    def encode_video(self, videos):
        return self.forward_video_features(self.forward_image_features(videos))

    def clamp_logit_scale(self):
        self.clip_model.clamp_logit_scale()
