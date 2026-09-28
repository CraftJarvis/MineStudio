'''
Date: 2024-11-25 07:03:41
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-07 14:21:06
FilePath: /MineStudio/minestudio/models/groot_one/body.py
'''
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import numpy as np
from einops import rearrange, repeat
from typing import List, Dict, Any, Tuple, Optional, Union
import av

import timm
from huggingface_hub import PyTorchModelHubMixin

from minestudio.models.base_policy import MinePolicy
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from minestudio.utils.register import Registers

def peel_off(item: Union[List, str]) -> str:
    """Recursively extracts a string from a potentially nested list of strings.

    If the item is a list, it calls itself with the first element of the list.
    If the item is a string, it returns the string.

    :param item: The item to peel, which can be a string or a list containing strings or other lists.
    :type item: Union[List, str]
    :returns: The innermost string found.
    :rtype: str
    """
    if isinstance(item, List):
        return peel_off(item[0])
    return item

class LatentSpace(nn.Module):
    """A module for creating a latent space with mean and log variance.

    This module takes an input tensor, projects it to a mean (mu) and a
    log variance (log_var), and then samples from the resulting Gaussian
    distribution during training. During evaluation, it returns the mean.

    :param hiddim: The hidden dimension of the input and latent space.
    :type hiddim: int
    """

    def __init__(self, hiddim: int) -> None:
        """Initialize the LatentSpace module.

        :param hiddim: The hidden dimension for the linear layers.
        :type hiddim: int
        """
        super().__init__()
        self.encode_mu = nn.Linear(hiddim, hiddim)
        self.encode_log_var = nn.Linear(hiddim, hiddim)

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Samples from a Gaussian distribution defined by mu and log_var.

        :param mu: The mean of the Gaussian distribution.
        :type mu: torch.Tensor
        :param log_var: The logarithm of the variance of the Gaussian distribution.
        :type log_var: torch.Tensor
        :returns: A tensor sampled from the N(mu, exp(log_var)) distribution.
        :rtype: torch.Tensor
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute latent variable z, its mean mu, and log variance log_var.

        During training, z is sampled from the distribution. During evaluation, z is mu.

        :param x: The input tensor.
        :type x: torch.Tensor
        :returns: A dictionary containing:
            - 'mu' (torch.Tensor): The mean of the latent distribution.
            - 'log_var' (torch.Tensor): The log variance of the latent distribution.
            - 'z' (torch.Tensor): The sampled (training) or mean (evaluation) latent variable.
        :rtype: Dict[str, torch.Tensor]
        """
        mu = self.encode_mu(x)
        log_var = self.encode_log_var(x)
        if self.training:
            z = self.sample(mu, log_var)
        else:
            z = mu
        return { 'mu': mu, 'log_var': log_var, 'z': z }

class VideoEncoder(nn.Module):
    """Encodes a sequence of video frames into a latent distribution.

    It uses Transformer encoders for spatial pooling within frames and temporal
    encoding across frames, followed by a LatentSpace module to get a distribution.

    :param hiddim: The hidden dimension for the model.
    :type hiddim: int
    :param num_spatial_layers: Number of Transformer encoder layers for spatial pooling.
                               Defaults to 2.
    :type num_spatial_layers: int
    :param num_temporal_layers: Number of Transformer encoder layers for temporal encoding.
                                Defaults to 2.
    :type num_temporal_layers: int
    :param num_heads: Number of attention heads in Transformer layers. Defaults to 8.
    :type num_heads: int
    :param dropout: Dropout rate in Transformer layers. Defaults to 0.1.
    :type dropout: float
    """
    
    def __init__(
        self, 
        hiddim: int, 
        num_spatial_layers: int=2, 
        num_temporal_layers: int=2, 
        num_heads: int=8, 
        dropout: float=0.1
    ) -> None:
        """Initialize the VideoEncoder.

        :param hiddim: Hidden dimension.
        :type hiddim: int
        :param num_spatial_layers: Number of spatial Transformer layers. Defaults to 2.
        :type num_spatial_layers: int
        :param num_temporal_layers: Number of temporal Transformer layers. Defaults to 2.
        :type num_temporal_layers: int
        :param num_heads: Number of attention heads. Defaults to 8.
        :type num_heads: int
        :param dropout: Dropout probability. Defaults to 0.1.
        :type dropout: float
        """
        super().__init__()
        self.hiddim = hiddim
        self.pooling = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim,
                nhead=num_heads,
                dim_feedforward=hiddim*2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_spatial_layers, 
        )
        self.encode_video = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim,
                nhead=num_heads,
                dim_feedforward=hiddim*2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_temporal_layers
        )
        self.encode_dist = LatentSpace(hiddim)

    def forward(self, images: torch.Tensor) -> Dict:
        """Encodes a batch of video frames.

        :param images: A tensor of video frames with shape (b, t, c, h, w),
                       where b=batch_size, t=time_steps, c=channels, h=height, w=width.
        :type images: torch.Tensor
        :returns: A dictionary representing the latent distribution from `LatentSpace`,
                  containing 'mu', 'log_var', and 'z'.
        :rtype: Dict[str, torch.Tensor]
        """
        x = rearrange(images, 'b t c h w -> (b t) (h w) c')
        x = self.pooling(x)
        x = x.mean(dim=1) # (b t) c
        x = rearrange(x, '(b t) c -> b t c', b=images.shape[0])
        x = self.encode_video(x)
        x = x.mean(dim=1) # b c
        dist = self.encode_dist(x)
        return dist


class ImageEncoder(nn.Module):
    """Encodes a single image into a latent distribution.

    Uses a Transformer encoder for spatial pooling, followed by a LatentSpace module.

    :param hiddim: The hidden dimension for the model.
    :type hiddim: int
    :param num_layers: Number of Transformer encoder layers for pooling. Defaults to 2.
    :type num_layers: int
    :param num_heads: Number of attention heads in Transformer layers. Defaults to 8.
    :type num_heads: int
    :param dropout: Dropout rate in Transformer layers. Defaults to 0.1.
    :type dropout: float
    """
    
    def __init__(self, hiddim: int, num_layers: int=2, num_heads: int=8, dropout: float=0.1) -> None:
        """Initialize the ImageEncoder.

        :param hiddim: Hidden dimension.
        :type hiddim: int
        :param num_layers: Number of Transformer layers. Defaults to 2.
        :type num_layers: int
        :param num_heads: Number of attention heads. Defaults to 8.
        :type num_heads: int
        :param dropout: Dropout probability. Defaults to 0.1.
        :type dropout: float
        """
        super().__init__()
        self.hiddim = hiddim
        self.pooling = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = hiddim,
                nhead = num_heads,
                dim_feedforward = hiddim*2,
                dropout = dropout,
                batch_first=True
            ),
            num_layers = num_layers, 
        )
        self.encode_dist = LatentSpace(hiddim)

    def forward(self, image: torch.Tensor) -> Dict:
        """Encodes a batch of images.

        :param image: A tensor of images with shape (b, c, h, w),
                      where b=batch_size, c=channels, h=height, w=width.
        :type image: torch.Tensor
        :returns: A dictionary representing the latent distribution from `LatentSpace`,
                  containing 'mu', 'log_var', and 'z'.
        :rtype: Dict[str, torch.Tensor]
        """
        x = rearrange(image, 'b c h w -> b (h w) c')
        x = self.pooling(x)
        x = x.mean(dim=1) # b c
        dist = self.encode_dist(x)
        return dist


class Decoder(nn.Module):
    """Decodes a sequence of latent vectors using a recurrent Transformer architecture.

    This module is typically used to generate sequences for policy and value estimation.

    :param hiddim: The hidden dimension of the model.
    :type hiddim: int
    :param num_heads: Number of attention heads in the recurrent Transformer blocks.
                      Defaults to 8.
    :type num_heads: int
    :param num_layers: Number of recurrent Transformer blocks. Defaults to 4.
    :type num_layers: int
    :param timesteps: The number of timesteps the recurrent model processes at once.
                      Defaults to 128.
    :type timesteps: int
    :param mem_len: The length of the memory used by the causal attention mechanism.
                    Defaults to 128.
    :type mem_len: int
    """
    
    def __init__(
        self, 
        hiddim: int, 
        num_heads: int = 8,
        num_layers: int = 4, 
        timesteps: int = 128, 
        mem_len: int = 128, 
    ) -> None:
        """Initialize the Decoder.

        :param hiddim: Hidden dimension.
        :type hiddim: int
        :param num_heads: Number of attention heads. Defaults to 8.
        :type num_heads: int
        :param num_layers: Number of recurrent layers. Defaults to 4.
        :type num_layers: int
        :param timesteps: Number of timesteps for recurrence. Defaults to 128.
        :type timesteps: int
        :param mem_len: Memory length for attention. Defaults to 128.
        :type mem_len: int
        """
        super().__init__()
        self.hiddim = hiddim
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
            attention_memory_size=mem_len + timesteps,
            n_block=num_layers,
        )
        self.lastlayer = FanInInitReLULayer(hiddim, hiddim, layer_type="linear", batch_norm=False, layer_norm=True)
        self.final_ln = nn.LayerNorm(hiddim)

    def forward(self, x: torch.Tensor, memory: List) -> Tuple[torch.Tensor, List]:
        """Forward pass of the Decoder.

        Processes the input sequence `x` using the recurrent Transformer blocks,
        updating the `memory` (recurrent state).

        :param x: Input tensor of shape (b, t, c), where b=batch_size, t=sequence_length, c=features.
        :type x: torch.Tensor
        :param memory: The recurrent state from the previous step. If None, an initial state is created.
        :type memory: List[torch.Tensor]
        :returns: A tuple containing:
            - x (torch.Tensor): The output tensor of shape (b, t, c).
            - memory (List[torch.Tensor]): The updated recurrent state.
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """
        b, t = x.shape[:2]
        if not hasattr(self, 'first'):
            self.first = torch.tensor([[False]], device=x.device).repeat(b, t)
        if memory is None:
            memory = [state.to(x.device) for state in self.recurrent.initial_state(b)]
        x, memory = self.recurrent(x, self.first, memory)
        x = F.relu(x, inplace=False)
        x = self.lastlayer(x)
        x = self.final_ln(x)
        return x, memory

    def initial_state(self, batch_size: int = None) -> List[torch.Tensor]:
        """Returns the initial recurrent state for the decoder.

        :param batch_size: The batch size for the initial state. If None, returns state for batch_size=1.
                           Defaults to None.
        :type batch_size: Optional[int]
        :returns: A list of tensors representing the initial recurrent state, moved to the model's device.
        :rtype: List[torch.Tensor]
        """
        device = next(self.parameters()).device
        if batch_size is None:
            return [t.squeeze(0).to(device) for t in self.recurrent.initial_state(1)]
        return [t.to(device) for t in self.recurrent.initial_state(batch_size)]

@Registers.model.register
class GrootPolicy(MinePolicy, PyTorchModelHubMixin):
    """GrootPolicy model for Minecraft, combining visual encoders and a recurrent decoder.

    This policy uses a pre-trained backbone (e.g., EfficientNet, ViT) to extract features
    from images. It has separate encoders for video sequences (reference trajectory)
    and single images (current observation). The features are fused and then processed
    by a recurrent decoder to produce policy and value outputs.

    :param backbone: Name of the timm model to use as a backbone (e.g., 'efficientnet_b0.ra_in1k').
    :type backbone: str
    :param freeze_backbone: Whether to freeze the weights of the pre-trained backbone. Defaults to True.
    :type freeze_backbone: bool
    :param hiddim: Hidden dimension for the policy network. Defaults to 1024.
    :type hiddim: int
    :param video_encoder_kwargs: Keyword arguments for the `VideoEncoder`. Defaults to {}.
    :type video_encoder_kwargs: Dict
    :param image_encoder_kwargs: Keyword arguments for the `ImageEncoder`. Defaults to {}.
    :type image_encoder_kwargs: Dict
    :param decoder_kwargs: Keyword arguments for the `Decoder`. Defaults to {}.
    :type decoder_kwargs: Dict
    :param action_space: The action space definition. Passed to `MinePolicy`.
    :type action_space: Optional[Any]
    """
    
    def __init__(
        self, 
        backbone: str='efficientnet_b0.ra_in1k', 
        freeze_backbone: bool=True,
        hiddim: int=1024,
        video_encoder_kwargs: Dict={},
        image_encoder_kwargs: Dict={},
        decoder_kwargs: Dict={},
        action_space=None,
    ):
        """Initialize the GrootPolicy.

        :param backbone: Name of the timm backbone model. Defaults to 'efficientnet_b0.ra_in1k'.
        :type backbone: str
        :param freeze_backbone: Whether to freeze backbone weights. Defaults to True.
        :type freeze_backbone: bool
        :param hiddim: Hidden dimension. Defaults to 1024.
        :type hiddim: int
        :param video_encoder_kwargs: Kwargs for VideoEncoder. Defaults to {}.
        :type video_encoder_kwargs: Dict
        :param image_encoder_kwargs: Kwargs for ImageEncoder. Defaults to {}.
        :type image_encoder_kwargs: Dict
        :param decoder_kwargs: Kwargs for Decoder. Defaults to {}.
        :type decoder_kwargs: Dict
        :param action_space: Action space definition. Defaults to None.
        :type action_space: Optional[Any]
        """
        super().__init__(hiddim=hiddim, action_space=action_space)
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        ])
        num_features = self.backbone.feature_info[-1]['num_chs']
        self.updim = nn.Conv2d(num_features, hiddim, kernel_size=1)
        self.video_encoder = VideoEncoder(hiddim, **video_encoder_kwargs)
        self.image_encoder = ImageEncoder(hiddim, **image_encoder_kwargs)
        self.decoder = Decoder(hiddim, **decoder_kwargs)
        self.timesteps = decoder_kwargs['timesteps']
        self.fuser = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim, 
                nhead=8, 
                dim_feedforward=hiddim*2, 
                dropout=0.1,
                batch_first=True
            ), 
            num_layers=2,
        )
        if freeze_backbone:
            print("Freezing backbone for GrootPolicy.")
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.condition_cache = {} # for infernce mode, to memory the generated conditions

    def encode_video(self, ref_video_path: str, resolution: Tuple[int, int] = (224, 224)) -> Dict:
        """Encodes a reference video from a file path into prior and posterior latent distributions.

        Reads a video file, extracts frames, preprocesses them using the backbone,
        and then uses the VideoEncoder and ImageEncoder to get latent distributions.

        :param ref_video_path: Path to the reference video file.
        :type ref_video_path: str
        :param resolution: Target resolution (width, height) to reformat video frames. Defaults to (224, 224).
        :type resolution: Tuple[int, int]
        :returns: A dictionary containing:
            - 'posterior_dist' (Dict): Latent distribution from the VideoEncoder.
            - 'prior_dist' (Dict): Latent distribution from the ImageEncoder (using the first frame).
        :rtype: Dict[str, Dict[str, torch.Tensor]]
        """
        frames = []
        # ref_video_path = ref_video_path[0][0] # unbatchify

        with av.open(ref_video_path, "r") as container:
            for fid, frame in enumerate(container.decode(video=0)):
                frame = frame.reformat(width=resolution[0], height=resolution[1]).to_ndarray(format="rgb24")
                frames.append(frame)

        reference = torch.from_numpy(np.stack(frames[:self.timesteps], axis=0) ).unsqueeze(0).to(self.device)
        reference = rearrange(reference, 'b t h w c -> (b t) c h w')
        reference = self.transforms(reference)
        reference = self.backbone(reference)[-1]
        reference = self.updim(reference)
        reference = rearrange(reference, '(b t) c h w -> b t c h w', b=1)
        posterior_dist = self.video_encoder(reference)
        prior_dist = self.image_encoder(reference[:, 0])

        posterior_dist['z'] = posterior_dist['z'].unsqueeze(0)
        prior_dist['z'] = prior_dist['z'].unsqueeze(0)

        print(
            "=======================================================\n"
            f"Ref video is from: {ref_video_path};\n"
            f"Num frames: {len(frames)}. \n"
            "=======================================================\n"
        )

        print(f"[📚] latent shape: {posterior_dist['z'].shape} | mean: {posterior_dist['z'].mean().item(): .3f} | std: {posterior_dist['z'].std(): .3f}")

        condition = {
            "posterior_dist": posterior_dist,
            "prior_dist": prior_dist
        }

        return condition

    def forward(self, input: Dict, memory: Optional[List[torch.Tensor]] = None) -> Dict:
        """Forward pass of the GrootPolicy.

        Processes the current image observation. If a `ref_video_path` is provided in the input
        (inference mode), it encodes the reference video (or uses a cached encoding) to get
        a condition `z`. If not (training mode), `z` is derived from the current batch of images.
        The image features and `z` are fused and passed to the decoder to get policy and value outputs.

        :param input: A dictionary of inputs. Expected to contain:
            - 'image' (torch.Tensor): Current image observations (b, t, h, w, c).
            - 'ref_video_path' (Optional[str] or Optional[List[str]]): Path to a reference video for conditioning (inference).
        :type input: Dict
        :param memory: The recurrent state for the decoder. Defaults to None (initial state will be used).
        :type memory: Optional[List[torch.Tensor]]
        :returns: A tuple containing:
            - latents (Dict): A dictionary with 'pi_logits', 'vpred', 'posterior_dist', and 'prior_dist'.
            - memory (List[torch.Tensor]): The updated recurrent state from the decoder.
        :rtype: Tuple[Dict[str, Any], List[torch.Tensor]]
        """
        b, t = input['image'].shape[:2]

        image = rearrange(input['image'], 'b t h w c -> (b t) c h w')
        image = self.transforms(image)
        image = self.backbone(image)[-1]
        image = self.updim(image)
        image = rearrange(image, '(b t) c h w -> b t c h w', b=b)

        if 'ref_video_path' in input:
            # input has `ref_video_path`, means inference mode
            ref_video_path = peel_off(input['ref_video_path'])
            if ref_video_path not in self.condition_cache:
                self.condition_cache[ref_video_path] = self.encode_video(ref_video_path)
            condition = self.condition_cache[ref_video_path]
            posterior_dist = condition['posterior_dist']
            prior_dist = condition['prior_dist']
            z = posterior_dist['z']
        else:
            # self-supervised training
            reference = image
            posterior_dist = self.video_encoder(reference)
            prior_dist = self.image_encoder(reference[:, 0])
            z = repeat(posterior_dist['z'], 'b c -> (b t) 1 c', t=t)

        x = rearrange(image, 'b t c h w -> (b t) (h w) c')
        x = torch.cat([x, z], dim=1)
        x = self.fuser(x)
        x = x.mean(dim=1) # (b t) c
        x = rearrange(x, '(b t) c -> b t c', b=b)
        x, memory = self.decoder(x, memory)
        pi_h = v_h = x
        pi_logits = self.pi_head(pi_h)
        vpred = self.value_head(v_h)
        latents = {
            "pi_logits": pi_logits, 
            "vpred": vpred, 
            "posterior_dist": posterior_dist, 
            "prior_dist": prior_dist
        }
        return latents, memory

    def initial_state(self, *args, **kwargs) -> Any:
        """Returns the initial recurrent state for the policy (from the decoder).

        :param args: Positional arguments passed to the decoder's `initial_state` method.
        :param kwargs: Keyword arguments passed to the decoder's `initial_state` method.
        :returns: The initial recurrent state.
        :rtype: Any
        """
        return self.decoder.initial_state(*args, **kwargs)

@Registers.model_loader.register
def load_groot_policy(ckpt_path: str = None):
    """Loads a GrootPolicy model.

    If `ckpt_path` is provided, it loads the model from the checkpoint.
    Otherwise, it loads a pre-trained model from Hugging Face Hub.

    :param ckpt_path: Path to a .ckpt model checkpoint file. Defaults to None.
    :type ckpt_path: Optional[str]
    :returns: The loaded GrootPolicy model.
    :rtype: GrootPolicy
    """
    if ckpt_path is None:
        repo_id = "CraftJarvis/MineStudio_GROOT.18w_EMA"
        return GrootPolicy.from_pretrained("CraftJarvis/MineStudio_GROOT.18w_EMA")

    ckpt = torch.load(ckpt_path)
    model = GrootPolicy(**ckpt['hyper_parameters']['model'])
    state_dict = {k.replace('mine_policy.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict, strict=True)
    return model

if __name__ == '__main__':
    model = load_groot_policy()
    model = GrootPolicy(
        backbone='timm/vit_base_patch32_clip_224.openai', 
        hiddim=1024,
        freeze_backbone=False,
        video_encoder_kwargs=dict(
            num_spatial_layers=2,
            num_temporal_layers=4,
            num_heads=8,
            dropout=0.1
        ),
        image_encoder_kwargs=dict(
            num_layers=2,
            num_heads=8,
            dropout=0.1
        ),
        decoder_kwargs=dict(
            num_layers=4,
            timesteps=128,
            mem_len=128
        )
    ).to("cuda")
    memory = None
    input = {
        'image': torch.zeros((2, 128, 224, 224, 3), dtype=torch.uint8).to("cuda"),
    }
    output, memory = model(input, memory)