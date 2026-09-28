'''
author:        caishaofei-MUS2 <1744260356@qq.com>
date:          2023-05-05 15:44:33
Copyright © Team CraftJarvis All rights reserved
'''
from copy import deepcopy
from email import policy
from typing import (
    List, Dict, Optional, Callable
)

import clip
import logging
import numpy as np
import torch
import gymnasium as gym
from gym3.types import DictType
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from efficientnet_pytorch import EfficientNet

from torch import _dynamo
# enable debug prints
torch._dynamo.config.log_level=logging.INFO
torch._dynamo.config.verbose=True

from minestudio.utils.vpt_lib.action_head import make_action_head
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.utils.vpt_lib.impala_cnn import ImpalaCNN
from minestudio.utils.vpt_lib.normalize_ewma import NormalizeEwma
from minestudio.utils.vpt_lib.scaled_mse_head import ScaledMSEHead
from minestudio.utils.vpt_lib.tree_util import tree_map
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from minestudio.utils.vpt_lib.misc import transpose

from minestudio.utils.factory import ( 
    build_backbone, 
    build_condition_embedding_layer, 
    build_conditioning_fusion_layer, 
    build_auxiliary_heads, 
    ActionEmbedding, 
    PastObsFusion, 
) 

class ScalableMinecraftPolicy(nn.Module):
    """
    :param recurrence_type:
        None                - No recurrence, adds no extra layers
        lstm                - (Depreciated). Singular LSTM
        multi_layer_lstm    - Multi-layer LSTM. Uses n_recurrence_layers to determine number of consecututive LSTMs
            Does NOT support ragged batching
        multi_masked_lstm   - Multi-layer LSTM that supports ragged batching via the first vector. This model is slower
            Uses n_recurrence_layers to determine number of consecututive LSTMs
        transformer         - Dense transformer
    :param init_norm_kwargs: kwargs for all FanInInitReLULayers.
    """

    def __init__(  
        self,
        recurrence_type="lstm",
        # backbone_type="IMPALA",
        obs_processing_width=256,
        hidsize=512,
        # single_output=False,  # True if we don't need separate outputs for action/value outputs
        init_norm_kwargs={},
        # Unused argument assumed by forc.
        input_shape=None,  # pylint: disable=unused-argument
        active_reward_monitors=None,      
        img_statistics=None,
        first_conv_norm=False,
        attention_mask_style="clipped_causal",
        attention_heads=8,
        attention_memory_size=2048,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        n_recurrence_layers=1,
        recurrence_is_residual=True,
        timesteps=None,
        use_pre_lstm_ln=True,  # Not needed for transformer
        # below are custimized arguments
        condition_embedding=None,
        action_embedding=None,
        conditioning_fusion=None,
        past_obs_fusion=None,
        action_space=None,
        backbone_kwargs={},
        auxiliary_backbone_kwargs={},
        condition_before_vision=False, 
        **unused_kwargs,
    ):
        super().__init__()
        assert recurrence_type in [
            "multi_layer_lstm",
            "multi_layer_bilstm",
            "multi_masked_lstm",
            "transformer",
            "none",
        ]

        active_reward_monitors = active_reward_monitors or {}
        self.hidsize = hidsize
        self.timesteps = timesteps
        # self.single_output = single_output
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True
        
        backbone_kwargs = {**backbone_kwargs, **unused_kwargs}
        backbone_kwargs['hidsize'] = hidsize
        backbone_kwargs['init_norm_kwargs'] = init_norm_kwargs
        backbone_kwargs['dense_init_norm_kwargs'] = self.dense_init_norm_kwargs
        # backbone_kwargs['require_goal_embedding'] = diff_mlp_embedding
        result_modules = build_backbone(**backbone_kwargs)
        self.img_preprocess = result_modules['preprocessing']
        self.img_process = result_modules['obsprocessing']

        self.pre_lstm_ln = nn.LayerNorm(hidsize) if use_pre_lstm_ln else None

        # build auxiliary backbones
        self.auxiliary_backbone_kwargs = auxiliary_backbone_kwargs
        if self.auxiliary_backbone_kwargs.get('enable', False):
            aux_result_modules = build_backbone(
                hidsize=hidsize,
                **self.auxiliary_backbone_kwargs, 
            )
            self.aux_img_preprocess = aux_result_modules['preprocessing']
            self.aux_img_process = aux_result_modules['obsprocessing']
        
        self.condition_before_vision = condition_before_vision

        self.condition_embedding = condition_embedding
        self.condition_embedding_layer = build_condition_embedding_layer(
            hidsize=hidsize,
            **self.condition_embedding
        ) if self.condition_embedding else None
        
        self.conditioning_fusion = conditioning_fusion
        self.conditioning_fusion_layer = build_conditioning_fusion_layer(
            hidsize=hidsize,
            **self.conditioning_fusion, 
        ) if self.conditioning_fusion else None
        
        self.action_embedding = action_embedding
        self.action_embedding_layer = ActionEmbedding(
            num_channels=hidsize, 
            action_space=action_space, 
            **self.action_embedding
        ) if self.action_embedding else None
        
        self.past_obs_fusion = past_obs_fusion
        self.past_obs_fusion_layer = PastObsFusion(
            hidsize=hidsize, 
            **self.past_obs_fusion, 
        ) if self.past_obs_fusion else None 
        
        self.recurrence_type = recurrence_type

        # The recurrent layer is implemented by OpenAI
        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type=recurrence_type,
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers,
        ) 

        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.final_ln = torch.nn.LayerNorm(hidsize)

        self.prepare_tensors()

    def prepare_tensors(self):
        MAT = []
        MAX_LENGTH = self.timesteps
        for i in range(MAX_LENGTH):
            row = []
            for j in range(MAX_LENGTH):
                row += [(j % (MAX_LENGTH-i)) + i]
            MAT += [row]
        self.MAT = torch.tensor(MAT)

    def output_latent_size(self):
        return self.hidsize

    def extract_vision_feats(
        self, 
        obs_preprocess: Callable, 
        obs_process: Callable,
        obs: Dict, 
        ce_latent: Optional[torch.Tensor] = None, 
        enable_past_fusion: bool = False,
    ) -> torch.Tensor:
        '''
        Extract vision features from the input image sequence. 
        The image sequence can be current episode's obsercations or 
        trajectoty of past observations (reused to encode trajectory), 
        in such case, the pre_cond should be `None`. 
        Also, you can specify whether to use the past observations. 
        args:
            obs_preprocess: Callable, preprocess the input image.
            obs_process: Callable, process the input image.
            obs['past_img']: (B, num_past, C, H, W) or Optional None 
            obs['img']: (B, T, C, H, W) 
            ce_latent: (B, T, C) 
            enable_past_fusion: bool, whether to use past observations. 
        return: 
            vision_feats: (B, T, C) or (B, T, C, X, X) or 
                          (B, T + num_past, C) or (B, T + num_past, C, X, X)
        '''
        if enable_past_fusion:
            assert obs['past_img'].shape[1] == self.past_obs_fusion['num_past_obs'], \
                f"past_img.len: {obs['past_img'].shape[1]} != num_past_obs: {self.past_obs_fusion['num_past_obs']}"
            img = torch.cat([obs['past_img'], obs['img']], dim=1)
            if ce_latent is not None:
                ce_latent = torch.cat([ torch.zeros_like(ce_latent[:, obs['past_img'].shape[1], :]), ce_latent ], dim=1)
        else:
            img = obs['img']
        B, T = img.shape[:2]
        x = obs_preprocess(img)
        vision_feats = obs_process(x, cond=ce_latent)
        vision_feats = vision_feats.reshape((B, T) + vision_feats.shape[2:])
        
        return vision_feats

    def forward(self, obs, state_in, context, ice_latent=None):
        '''
        Args:
            obs: Dict, observation. 
            state_in: Dict, input state. 
            ice_latent: Optional[torch.Tensor], input condition embedding. 
                For example, in the inference stage, the condition embedding 
                is encoded before running this forward function. Then it 
                should use ice_latent argument. 
        '''
        T = obs['img'].shape[1]
        result_latents = {}
        # This is only for using the Impala FiLM backbone. 
        # Otherwise, the following codes won't be executed. 
        if ice_latent is None:
            if self.condition_before_vision:
                ce_latent = self.condition_embedding_layer(
                    texts=obs.get('text'), 
                    device=obs['img'].device
                )
                ce_latent = ce_latent.unsqueeze(1).expand(-1, obs['img'].shape[1], -1)
            else:
                ce_latent = None
        else:
            ice_latent = ice_latent.unsqueeze(1).expand(-1, T, -1)
            ce_latent = ice_latent

        # Extract vision features from the input image sequence. 
        # The result feature may be a 3D tensor or a 5D tensor. 
        vi_latent = self.extract_vision_feats(
            obs_preprocess=self.img_preprocess, 
            obs_process=self.img_process, 
            obs=obs, 
            ce_latent=ce_latent, 
            enable_past_fusion=False, 
        )
        
        # Extract auxiliary vision features from the input image sequence. 
        if self.auxiliary_backbone_kwargs.get('enable', False):
            av_latent = self.extract_vision_feats(
                obs_preprocess=self.aux_img_preprocess, 
                obs_process=self.aux_img_process,
                obs=obs, 
                enable_past_fusion=True, 
            )
        else:
            av_latent = None
        
        # Compute the condition embeddings. 
        # The condition type can be text, subgoal, and trajectory. 
        if ice_latent is None:
            if (
                getattr(self, 'condition_embedding_layer', None) 
                and not 
                self.condition_before_vision 
            ):
                goal_kwargs = {
                    'vision_feats': vi_latent, 
                    'texts': obs.get('text', None), 
                    'subgoal': obs.get('subgoal', {}), 
                }
                cond_ret = self.condition_embedding_layer(**goal_kwargs)
                # Here, the condition return can be tensor or distributions. 
                if isinstance(cond_ret, torch.Tensor):
                    ce_latent = cond_ret
                    if len(ce_latent.shape) == 2:
                        ce_latent = ce_latent.unsqueeze(1)
                    if ce_latent.shape[1] == 1:
                        ce_latent = ce_latent.expand(-1, obs['img'].shape[1], -1)
                elif isinstance(cond_ret, List) and isinstance(cond_ret[0], torch.distributions.Distribution):
                    result_latents['condition_dists'] = cond_ret
                    ce_latent = torch.stack([d.rsample() for d in cond_ret], dim=1)
                    # # random replace item with the future embedding. 
                    # perm = torch.randperm(ce_latent.shape[1]).unsqueeze(1)
                    # index = (
                    #     torch.gather(self.MAT, 1, perm)
                    #     .squeeze(1)
                    # )
                    # ce_latent = ce_latent[:, index, :]
        else:
            ce_latent = ice_latent
        
        # Use the condition embeddings to condition the vision features. 
        if getattr(self, 'conditioning_fusion_layer', None): 
            vi_latent = self.conditioning_fusion_layer(vi_latent, ce_latent) 

        # ov is the original vision features. 
        # oa is the original auxiliary vision features. 
        if getattr(self, 'past_obs_fusion_layer', None):
            if av_latent is not None and av_latent.shape[1] > T:
                oa_latent = av_latent[:, -T:, ...]
                av_latent = self.past_obs_fusion_layer(av_latent)
            else:
                oa_latent = av_latent
            
            if vi_latent.shape[1] > T:
                ov_latent = vi_latent[:, -T:, ...]
                vi_latent = self.past_obs_fusion_layer(vi_latent)
            else:
                ov_latent = vi_latent
                
        else:
            ov_latent = vi_latent 
            oa_latent = av_latent 
        
        # Here, we use the original vision features for decision making. 
        x = ov_latent
        
        # Inject previous action embeddings. 
        if getattr(self, 'action_embedding_layer', None):
            prev_action_embedding = self.action_embedding_layer(obs["prev_action"])
            x = prev_action_embedding + x

        # Won't be executed in the case of transformer.
        if self.pre_lstm_ln is not None:
            x = self.pre_lstm_ln(x)
        
        if self.recurrent_layer is not None:
            x, state_out = self.recurrent_layer(x, context["first"], state_in)
        else:
            state_out = state_in
        
        tf_latent = x
        x = F.relu(x, inplace=False)
        
        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x
        
        # Return intermediate latents for decision making and other auxiliary tasks. 
        result_latents.update({
            "vi_latent": vi_latent,
            "ov_latent": ov_latent,
            "av_latent": av_latent,
            "oa_latent": oa_latent,
            "pi_latent": pi_latent,
            "vf_latent": vf_latent,
            "tf_latent": tf_latent,
            "ce_latent": ce_latent,
        })
        
        return result_latents, state_out

    def initial_state(self, batchsize):
        if self.recurrent_layer:
            return self.recurrent_layer.initial_state(batchsize)
        else:
            return None


class MinecraftAgentPolicy(nn.Module):
    
    def __init__(self, action_space, policy_kwargs, pi_head_kwargs, auxiliary_head_kwargs):
        super().__init__()
        self.net = ScalableMinecraftPolicy(**policy_kwargs, action_space=action_space)

        self.action_space = action_space

        self.value_head = self.make_value_head(self.net.output_latent_size())
        self.pi_head = self.make_action_head(self.net.output_latent_size(), **pi_head_kwargs)
        self.auxiliary_heads = nn.ModuleDict(build_auxiliary_heads(
            auxiliary_head_kwargs=auxiliary_head_kwargs, 
            hidsize=policy_kwargs['hidsize'],
        ))

    def make_value_head(self, v_out_size: int, norm_type: str = "ewma", norm_kwargs: Optional[Dict] = None):
        return ScaledMSEHead(v_out_size, 1, norm_type=norm_type, norm_kwargs=norm_kwargs)

    def make_action_head(self, pi_out_size: int, **pi_head_opts):
        return make_action_head(self.action_space, pi_out_size, **pi_head_opts)

    def initial_state(self, batch_size: int):
        return self.net.initial_state(batch_size)

    def reset_parameters(self):
        super().reset_parameters()
        self.net.reset_parameters()
        self.pi_head.reset_parameters()
        self.value_head.reset_parameters()

    def forward(
        self, 
        obs: Dict, 
        first: torch.Tensor, 
        state_in: List[torch.Tensor], 
        stage: str = 'train', 
        ice_latent: Optional[torch.Tensor] = None, 
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if isinstance(obs, dict):
            # We don't want to mutate the obs input.
            obs = obs.copy()
            mask = obs.pop("mask", None)
        else:
            mask = None
        
        latents, state_out = self.net(
            obs=obs, 
            state_in=state_in, 
            context={"first": first}, 
            ice_latent=ice_latent,
        )
        result = {
            'pi_logits': self.pi_head(latents['pi_latent']),
            'vpred': self.value_head(latents['vf_latent']),
        }
        
        for head, module in self.auxiliary_heads.items():
            result[head] = module(latents, stage=stage)
        
        return result, state_out, latents