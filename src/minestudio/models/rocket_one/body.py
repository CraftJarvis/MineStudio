'''
Date: 2024-11-10 15:52:16
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-15 17:08:36
FilePath: /MineStudio/minestudio/models/rocket_one/body.py
'''
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from einops import rearrange
from typing import List, Dict, Any, Tuple, Optional

import timm
from huggingface_hub import PyTorchModelHubMixin
from minestudio.models.base_policy import MinePolicy
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from minestudio.utils.register import Registers

@Registers.model.register
class RocketPolicy(MinePolicy, PyTorchModelHubMixin):
    """RocketPolicy model for Minecraft, using a Vision Transformer (ViT) backbone.

    This policy processes an RGB image concatenated with an object mask. It uses a
    pre-trained ViT backbone for feature extraction, followed by Transformer-based
    pooling and recurrent blocks for temporal processing. It also incorporates an
    embedding for interaction types.

    :param backbone: Name of the timm model to use as a backbone (e.g., 'timm/vit_base_patch16_224.dino').
                     Defaults to 'timm/vit_base_patch16_224.dino'.
    :type backbone: str
    :param hiddim: Hidden dimension for the policy network. Defaults to 1024.
    :type hiddim: int
    :param num_heads: Number of attention heads in Transformer layers. Defaults to 8.
    :type num_heads: int
    :param num_layers: Number of recurrent Transformer blocks. Defaults to 4.
    :type num_layers: int
    :param timesteps: The number of timesteps the recurrent model processes at once. Defaults to 128.
    :type timesteps: int
    :param mem_len: The length of the memory used by the causal attention mechanism in recurrent blocks.
                    Defaults to 128.
    :type mem_len: int
    :param action_space: The action space definition. Passed to `MinePolicy`. Defaults to None.
    :type action_space: Optional[Any]
    :param nucleus_prob: Nucleus probability for sampling actions. Passed to `MinePolicy`.
                         Defaults to 0.85.
    :type nucleus_prob: float
    """
    
    def __init__(self, 
        backbone: str = 'timm/vit_base_patch16_224.dino', 
        hiddim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 4,
        timesteps: int = 128,
        mem_len: int = 128,
        action_space = None,
        nucleus_prob = 0.85,
    ):
        """Initialize the RocketPolicy.

        :param backbone: Name of the timm backbone model. Defaults to 'timm/vit_base_patch16_224.dino'.
        :type backbone: str
        :param hiddim: Hidden dimension. Defaults to 1024.
        :type hiddim: int
        :param num_heads: Number of attention heads. Defaults to 8.
        :type num_heads: int
        :param num_layers: Number of recurrent layers. Defaults to 4.
        :type num_layers: int
        :param timesteps: Number of timesteps for recurrence. Defaults to 128.
        :type timesteps: int
        :param mem_len: Memory length for attention. Defaults to 128.
        :type mem_len: int
        :param action_space: Action space definition. Defaults to None.
        :type action_space: Optional[Any]
        :param nucleus_prob: Nucleus probability for sampling. Defaults to 0.85.
        :type nucleus_prob: float
        """
        super().__init__(hiddim=hiddim, action_space=action_space, nucleus_prob=nucleus_prob)
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True, in_chans=4)
        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        ])
        num_features = self.backbone.feature_info[-1]['num_chs']
        self.updim = nn.Conv2d(num_features, hiddim, kernel_size=1, bias=False)
        self.pooling = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim, 
                nhead=num_heads, 
                dim_feedforward=hiddim*2, 
                dropout=0.1,
                batch_first=True
            ), 
            num_layers=2,
        )
        
        self.interaction = nn.Embedding(10, hiddim) # denotes the number of interaction types
        self.recurrent = ResidualRecurrentBlocks(
            hidsize=hiddim,
            timesteps=timesteps, 
            recurrence_type="transformer", 
            is_residual=True,
            use_pointwise_layer=True,
            pointwise_ratio=4, 
            pointwise_use_activation=False, 
            attention_mask_style="clipped_causal", 
            attention_heads=num_heads,
            attention_memory_size=mem_len+timesteps,
            n_block=num_layers,
            inject_condition=True, # inject obj_embedding as the condition
        )
        self.lastlayer = FanInInitReLULayer(hiddim, hiddim, layer_type="linear", batch_norm=False, layer_norm=True)
        self.final_ln = nn.LayerNorm(hiddim)

    def forward(self, input: Dict, memory: Optional[List[torch.Tensor]] = None) -> Dict:
        """Forward pass of the RocketPolicy.

        Processes the input image and segmentation mask, extracts features, and passes them
        through recurrent layers to produce policy and value predictions.

        Input dictionary is expected to contain:
        - 'image': (b, t, h, w, c) tensor of RGB images.
        - 'segment' or 'segmentation': Dictionary containing:
            - 'obj_mask': (b, t, h, w) tensor of object masks.
            - 'obj_id': (b, t) tensor of object interaction type IDs.

        :param input: Dictionary of input tensors.
        :type input: Dict
        :param memory: Optional list of recurrent state tensors. If None, an initial state is used.
        :type memory: Optional[List[torch.Tensor]]
        :returns: A tuple containing:
            - latents (Dict): Dictionary with 'pi_logits' and 'vpred'.
            - memory (List[torch.Tensor]): Updated list of recurrent state tensors.
        :rtype: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]
        """
        # import ipdb; ipdb.set_trace()
        ckey = 'segment' if 'segment' in input else 'segmentation'
        
        b, t = input['image'].shape[:2]
        rgb = rearrange(input['image'], 'b t h w c -> (b t) c h w')
        rgb = self.transforms(rgb)

        obj_mask = input[ckey]['obj_mask']
        obj_mask = rearrange(obj_mask, 'b t h w -> (b t) 1 h w')
        x = torch.cat([rgb, obj_mask], dim=1)
        feats = self.backbone(x)
        x = self.updim(feats[-1])
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.pooling(x).mean(dim=1) 
        x = rearrange(x, "(b t) c -> b t c", b=b)

        y = self.interaction(input[ckey]['obj_id'] + 1) # b t c
        if not hasattr(self, 'first'):
            self.first = torch.tensor([[False]], device=x.device).repeat(b, t)
        if memory is None:
            memory = [state.to(x.device) for state in self.recurrent.initial_state(b)]
        
        z, memory = self.recurrent(x, self.first, memory, ce_latent=y)
        
        z = F.relu(z, inplace=False)
        z = self.lastlayer(z)
        z = self.final_ln(z)
        pi_h = v_h = z
        pi_logits = self.pi_head(pi_h)
        vpred = self.value_head(v_h)
        latents = {"pi_logits": pi_logits, "vpred": vpred}
        return latents, memory

    def initial_state(self, batch_size: int = None) -> List[torch.Tensor]:
        """Returns the initial recurrent state for the policy.

        :param batch_size: The batch size for the initial state. If None, returns state for batch_size=1.
                           Defaults to None.
        :type batch_size: Optional[int]
        :returns: A list of tensors representing the initial recurrent state, moved to the model's device.
        :rtype: List[torch.Tensor]
        """
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.recurrent.initial_state(1)]
        return [t.to(self.device) for t in self.recurrent.initial_state(batch_size)]

@Registers.model_loader.register
def load_rocket_policy(ckpt_path: Optional[str] = None):
    """Loads a RocketPolicy model.

    If `ckpt_path` is provided, it loads the model from the checkpoint.
    Otherwise, it loads a pre-trained model from Hugging Face Hub.

    :param ckpt_path: Path to a .ckpt model checkpoint file. Defaults to None.
    :type ckpt_path: Optional[str]
    :returns: The loaded RocketPolicy model.
    :rtype: RocketPolicy
    """
    if ckpt_path is None:
        model = RocketPolicy.from_pretrained("CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
        return model
    ckpt = torch.load(ckpt_path)
    model = RocketPolicy(**ckpt['hyper_parameters']['model'])
    state_dict = {k.replace('mine_policy.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict, strict=True)
    return model

if __name__ == '__main__':
    # model = load_rocket_policy()
    model = RocketPolicy.from_pretrained("CraftJarvis/MineStudio_ROCKET-1.12w_EMA").to("cuda")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Params (MB): {num_params / 1e6 :.2f}")
    
    for key in ["backbone", "updim", "pooling", "interaction", "recurrent", "lastlayer", "final_ln"]:
        num_params = sum(p.numel() for p in getattr(model, key).parameters())
        print(f"{key} Params (MB): {num_params / 1e6 :.2f}")

    output, memory = model(
        input={
            'image': torch.zeros(1, 128, 224, 224, 3).to("cuda"), 
            'segment': {
                'obj_id': torch.zeros(1, 128, dtype=torch.long).to("cuda"),
                'obj_mask': torch.zeros(1, 128, 224, 224).to("cuda"),
            }
        }
    )
    print(output.keys())