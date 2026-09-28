from copy import deepcopy
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from minestudio.models.base_policy import MinePolicy
from huggingface_hub import PyTorchModelHubMixin

from minestudio.utils.mineclip_lib.mineclip import MineCLIP
from minestudio.utils.vpt_lib.impala_cnn import ImpalaCNN
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from minestudio.models.base_policy import MinePolicy
from minestudio.online.utils import auto_stack, auto_to_torch

class TranslatorVAE(torch.nn.Module):
    """
    Variational Autoencoder for translating between text and visual embeddings.
    
    This VAE learns to map text embeddings to visual embeddings through a latent space,
    enabling text-to-visual translation for multimodal learning tasks. The encoder
    takes concatenated visual and text embeddings to produce latent representations,
    while the decoder reconstructs visual embeddings from latent codes and text.
    """

    def __init__(self, input_dim=512, hidden_dim=256, latent_dim=256):
        """
        Initialize the TranslatorVAE with specified dimensions.
        
        :param input_dim: Dimension of input visual and text embeddings
        :param hidden_dim: Dimension of hidden layers in encoder and decoder
        :param latent_dim: Dimension of the latent space
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, visual_embeddings, text_embeddings):
        """
        Encode concatenated visual and text embeddings into latent parameters.
        
        :param visual_embeddings: Visual embedding tensor of shape (B, input_dim)
        :param text_embeddings: Text embedding tensor of shape (B, input_dim)
        :returns: Encoded tensor containing mean and log variance (B, 2*latent_dim)
        """
        # Concatenate the visual and text embeddings.
        x = torch.cat([visual_embeddings, text_embeddings], dim=1)
        # Encode the concatenated embeddings into a latent vector.
        return self.encoder(x)

    def sample(self, mu, logvar):
        """
        Sample a latent vector using the reparameterization trick.
        
        Applies the reparameterization trick to sample from a Gaussian distribution
        defined by mu and logvar, enabling backpropagation through stochastic sampling.
        
        :param mu: Mean tensor of shape (B, latent_dim)
        :param logvar: Log variance tensor of shape (B, latent_dim)
        :returns: Sampled latent vector of shape (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent_vector, text_embeddings):
        """
        Decode latent vector and text embeddings into visual embeddings.
        
        :param latent_vector: Latent representation tensor of shape (B, latent_dim)
        :param text_embeddings: Text embedding tensor of shape (B, input_dim)
        :returns: Reconstructed visual embeddings of shape (B, input_dim)
        """
        # Concatenate the latent vector and text embeddings.
        x = torch.cat([latent_vector, text_embeddings], dim=1)
        # Decode the concatenated embeddings into a visual embedding.
        return self.decoder(x)

    def forward(self, text_embeddings, deterministic=False):
        """
        Generate visual embeddings from text embeddings using prior distribution.
        
        Uses a zero-mean, unit-variance prior to sample latent vectors and decode
        them into visual embeddings conditioned on the input text embeddings.
        
        :param text_embeddings: Input text embeddings of shape (B, input_dim)
        :param deterministic: If True, use mean of prior; if False, sample from prior
        :returns: Generated visual embeddings of shape (B, input_dim)
        """
        # Use the prior as the mean and logvar.
        mu = torch.zeros(text_embeddings.shape[0], self.latent_dim).to(text_embeddings.device)
        logvar = torch.zeros(text_embeddings.shape[0], self.latent_dim).to(text_embeddings.device)

        # Sample a latent vector from the mu and logvar.
        if deterministic:
            latent_vector = mu
        else:
            latent_vector = self.sample(mu, logvar)

        # Decode the latent vector into a visual embedding.
        pred_visual_embeddings = self.decode(latent_vector, text_embeddings)

        return pred_visual_embeddings

class ImgPreprocessing(torch.nn.Module):
    """
    Image preprocessing module for normalization and scaling.
    
    Normalizes incoming images using either provided statistics (mean/std) or
    simple scaling. Supports both statistical normalization from pre-computed
    statistics and basic scaling by a constant factor.
    
    :param img_statistics: Path to npz file containing mean and std statistics.
        If specified, normalize images using these statistics.
    :param scale_img: If True and img_statistics not specified, scale images by 1/255.
    """

    def __init__(self, img_statistics: Optional[str] = None, scale_img: bool = True):
        """
        Initialize image preprocessing with normalization parameters.
        
        :param img_statistics: Optional path to npz file with mean/std statistics
        :param scale_img: Whether to scale images by 1/255 when no statistics provided
        """
        super().__init__()
        self.img_mean = None
        if img_statistics is not None:
            img_statistics = dict(**np.load(img_statistics))
            self.img_mean = torch.nn.Parameter(torch.Tensor(img_statistics["mean"]), requires_grad=False)
            self.img_std = torch.nn.Parameter(torch.Tensor(img_statistics["std"]), requires_grad=False)
        else:
            self.ob_scale = 255.0 if scale_img else 1.0

    def forward(self, img):
        """
        Apply preprocessing normalization to input images.
        
        :param img: Input image tensor of shape (B, C, H, W) or (B, T, C, H, W)
        :returns: Normalized image tensor with same shape as input
        """
        x = img
        if self.img_mean is not None:
            x = (x - self.img_mean) / self.img_std
        else:
            x = x / self.ob_scale
        return x

class ImgObsProcess(torch.nn.Module):
    """
    Image observation processing using ImpalaCNN followed by a linear layer.
    
    Processes image observations through an Impala CNN architecture and then
    applies a linear transformation to produce the final output embeddings.
    
    :param cnn_outsize: Output dimension of the Impala CNN
    :param output_size: Output size of the final linear layer
    :param dense_init_norm_kwargs: Initialization kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: Initialization kwargs for 2D and 3D conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        """
        Initialize image observation processing module.
        
        :param cnn_outsize: Output dimension of the CNN backbone
        :param output_size: Final output embedding dimension
        :param dense_init_norm_kwargs: Kwargs for dense layer initialization
        :param init_norm_kwargs: Kwargs for convolution layer initialization
        :param kwargs: Additional arguments passed to ImpalaCNN
        """
        super().__init__()
        self.cnn = ImpalaCNN(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

    def forward(self, img):
        """
        Process image through CNN and linear transformation.
        
        :param img: Input image tensor of shape (B, C, H, W)
        :returns: Processed image features of shape (B, output_size)
        """
        return self.linear(self.cnn(img))

class MinecraftPolicy(torch.nn.Module):
    """
    Neural network policy for Minecraft gameplay with multimodal inputs.
    
    This policy combines visual and textual information processing through CNN and 
    recurrent architectures. It supports various recurrence types including LSTM,
    masked LSTM, and transformer variants for sequential decision making.
    
    The architecture processes images through ImpalaCNN, combines with MineCLIP
    embeddings for text conditioning, and uses recurrent layers for temporal modeling.
    
    :param recurrence_type: Type of recurrent architecture:
        - 'multi_layer_lstm': Multi-layer LSTM (no ragged batching support)
        - 'multi_layer_bilstm': Bidirectional multi-layer LSTM  
        - 'multi_masked_lstm': Multi-layer LSTM with ragged batching support
        - 'transformer': Dense transformer architecture
        - 'none': No recurrence
    :param init_norm_kwargs: Initialization kwargs for all FanInInitReLULayers
    """

    def __init__(
        self,
        recurrence_type="lstm",
        impala_width=1,
        impala_chans=(16, 32, 32),
        obs_processing_width=256,
        hidsize=512,
        single_output=False,  # True if we don't need separate outputs for action/value outputs
        img_shape=None,
        scale_input_img=True,
        only_img_input=False,
        init_norm_kwargs={},
        impala_kwargs={},
        # Unused argument assumed by forc.
        input_shape=None,  # pylint: disable=unused-argument
        active_reward_monitors=None,
        img_statistics=None,
        first_conv_norm=False,
        diff_mlp_embedding=False,
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
        mineclip_embed_dim=512,  # MODIFIED (added this)
        **unused_kwargs,
    ):
        """
        Initialize the Minecraft policy network.
        
        :param recurrence_type: Type of recurrent layer ('multi_layer_lstm', 'multi_layer_bilstm', 'multi_masked_lstm', 'transformer', 'none')
        :param impala_width: Width multiplier for Impala CNN channels
        :param impala_chans: Base channel sizes for Impala CNN layers
        :param obs_processing_width: Width for observation processing layers
        :param hidsize: Hidden state size for recurrent layers and output
        :param single_output: Whether to use single output for both policy and value
        :param img_shape: Expected input image shape
        :param scale_input_img: Whether to scale input images by 1/255
        :param only_img_input: Whether to use only image inputs (unused)
        :param init_norm_kwargs: Initialization kwargs for normalization layers
        :param impala_kwargs: Additional kwargs for Impala CNN
        :param input_shape: Legacy parameter (unused)
        :param active_reward_monitors: Dictionary of active reward monitors
        :param img_statistics: Path to image normalization statistics
        :param first_conv_norm: Whether to apply normalization to first conv layer
        :param diff_mlp_embedding: Whether to use different MLP embedding (unused)
        :param attention_mask_style: Attention masking style for transformer
        :param attention_heads: Number of attention heads for transformer
        :param attention_memory_size: Memory size for attention mechanism
        :param use_pointwise_layer: Whether to use pointwise layers in transformer
        :param pointwise_ratio: Ratio for pointwise layer dimensions
        :param pointwise_use_activation: Whether to use activation in pointwise layers
        :param n_recurrence_layers: Number of recurrent layers to stack
        :param recurrence_is_residual: Whether recurrent layers use residual connections
        :param timesteps: Number of timesteps for sequence processing
        :param use_pre_lstm_ln: Whether to use layer norm before LSTM
        :param mineclip_embed_dim: Dimension of MineCLIP embeddings
        :param unused_kwargs: Additional unused arguments
        """
        super().__init__()
        assert recurrence_type in [
            "multi_layer_lstm",
            "multi_layer_bilstm",
            "multi_masked_lstm",
            "transformer",
            "none",
        ]

        active_reward_monitors = active_reward_monitors or {}

        self.single_output = single_output

        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.hidsize = hidsize

        # Dense init kwargs replaces batchnorm/groupnorm with layernorm
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        # Setup inputs
        self.img_preprocess = ImgPreprocessing(img_statistics=img_statistics, scale_img=scale_input_img)
        self.img_process = ImgObsProcess(
            cnn_outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            **impala_kwargs,
        )

        self.pre_lstm_ln = torch.nn.LayerNorm(hidsize) if use_pre_lstm_ln else None
        self.diff_obs_process = None

        self.recurrence_type = recurrence_type

        self.recurrent_layer = None
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

        # MODIFIED (added this)
        self.mineclip_embed_linear = torch.nn.Linear(mineclip_embed_dim, hidsize)

    def output_latent_size(self):
        """
        Get the size of the output latent representation.
        
        :returns: Integer size of the hidden/latent dimension
        """
        return self.hidsize

    def forward(self, ob, state_in, context):
        """
        Forward pass through the Minecraft policy network.
        
        Processes multimodal observations (images and MineCLIP embeddings) through
        CNN and recurrent layers to produce policy and value latent representations.
        
        :param ob: Observation dictionary containing 'img' and 'mineclip_embed' keys
        :param state_in: Input recurrent state from previous timestep
        :param context: Context dictionary containing 'first' episode flags
        :returns: Tuple of ((pi_latent, vf_latent), state_out) where latents are
                 policy and value representations and state_out is updated recurrent state
        """
        b, t = ob["img"].shape[:2]
        first = context["first"].bool()

        x = self.img_preprocess(ob["img"])
        x = self.img_process(x)

        if self.diff_obs_process:
            processed_obs = self.diff_obs_process(ob["diff_goal"])
            x = processed_obs + x

        # MODIFIED (added this)
        mineclip_embed = ob["mineclip_embed"].reshape(b * t, -1)
        # Normalize mineclip_embed (doesn't work because the norm is way too small then?)
        # mineclip_embed = F.normalize(mineclip_embed, dim=-1)
        mineclip_embed = self.mineclip_embed_linear(mineclip_embed)
        mineclip_embed = mineclip_embed.reshape(b, t, -1)
        x = x + mineclip_embed

        if self.pre_lstm_ln is not None:
            x = self.pre_lstm_ln(x)

        if self.recurrent_layer is not None:
            x, state_out = self.recurrent_layer(x, first, state_in)
        else:
            state_out = state_in

        x = F.relu(x, inplace=False)

        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x
        if self.single_output:
            return pi_latent, state_out
        return (pi_latent, vf_latent), state_out

    def initial_state(self, batchsize):
        """
        Initialize the recurrent state for a new episode.
        
        :param batchsize: Batch size for state initialization
        :returns: Initial recurrent state tensors or None if no recurrence
        """
        if self.recurrent_layer:
            return self.recurrent_layer.initial_state(batchsize)
        else:
            return None

class SteveOnePolicy(MinePolicy, PyTorchModelHubMixin):
    """
    Complete STEVE-1 policy combining visual processing, text conditioning, and action prediction.
    
    This policy integrates MineCLIP for multimodal understanding, a TranslatorVAE for 
    text-to-visual translation, and a MinecraftPolicy network for decision making.
    It supports both text and video conditioning through classifier-free guidance.
    
    The architecture enables text-conditioned gameplay by translating textual instructions
    into visual embeddings that guide the policy's behavior in Minecraft environments.
    """
    
    def __init__(
        self, 
        mineclip_kwargs: dict = {},
        prior_kwargs: dict = {},
        policy_kwargs: dict = {},
        freeze_mineclip: bool = True,
        action_space = None,
    ):
        """
        Initialize the STEVE-1 policy with all components.
        
        :param mineclip_kwargs: Configuration for MineCLIP multimodal encoder
        :param prior_kwargs: Configuration for TranslatorVAE prior model
        :param policy_kwargs: Configuration for MinecraftPolicy network
        :param freeze_mineclip: Whether to freeze MineCLIP parameters during training
        :param action_space: Action space specification for the environment
        """
        net = MinecraftPolicy(** policy_kwargs)
        super().__init__(hiddim=net.hidsize, action_space=action_space)

        self.net = net
        self.prior = TranslatorVAE(** prior_kwargs)
        self.mineclip = MineCLIP(** mineclip_kwargs)

        if freeze_mineclip:
            for param in self.mineclip.parameters():
                param.requires_grad = False

    def prepare_condition(self, instruction: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """
        Prepare conditioning information from text or video instructions.
        
        Processes either text instructions (via TranslatorVAE) or video demonstrations
        (via direct MineCLIP encoding) to create conditioning embeddings for the policy.
        
        :param instruction: Dictionary containing either 'text' or 'video' key plus 'cond_scale'
        :param deterministic: Whether to use deterministic sampling for text conditioning
        :returns: Dictionary with 'cond_scale' and 'mineclip_embeds' for policy conditioning
        :raises AssertionError: If instruction lacks required keys or has conflicting modalities
        """
        assert 'cond_scale' in instruction, "instruction must have 'cond_scale' key."

        if 'video' in instruction:
            assert 'text' not in instruction, "cannot have both text and video in instruction"
            
            video = instruction['video']

            if isinstance(video, np.ndarray):
                video = torch.from_numpy(video).to(self.device)
            if video.dim() == 4:
                video = rearrange(video, 'T H W C -> 1 T C H W')
            if video.shape[2] != 3:
                video = rearrange(video, 'B T H W C -> B T C H W')

            assert video.dim() == 5 and video.shape[2] == 3, "video must be a 5D tensor with shape (B, T, C, H, W) or (B, T, H, W, C)"
            B, T, C, H, W = video.shape

            if video.dtype == torch.uint8:
                mineclip_inputs = video.float()
            elif video.dtype == torch.float32:
                assert video.abs().max() <= 1.0, "float32 video must be in range [-1, 1]"
                mineclip_inputs = video * 255.0
            else:
                raise ValueError("video must be either uint8 or float32.")
            
            mineclip_inputs = rearrange(
                torch.nn.functional.interpolate(
                    rearrange(mineclip_inputs, 'B T C H W -> (B T) C H W'),
                    size=(160, 256), 
                    mode='bilinear', 
                    align_corners=False
                ),
                '(B T) C H W -> B T C H W',
                B=B, T=T
            )

            mineclip_embeds = self.mineclip.encode_video(mineclip_inputs)
        else:
            assert 'text' in instruction, "instruction must have either text or video."

            texts = instruction['text']
            if isinstance(texts[0], list):
                texts = texts[0]
            elif isinstance(texts, str):
                texts = [texts]
            assert isinstance(texts, list) and isinstance(texts[0], str), "text must be a string or a list of strings."
            
            text_embeds = self.mineclip.encode_text(texts)
            mineclip_embeds = self.prior(text_embeds, deterministic=deterministic)

        return {
            'cond_scale': instruction['cond_scale'],
            'mineclip_embeds': mineclip_embeds,
        }

    def forward(
        self, 
        input: Dict[str, Any], 
        state_in: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with classifier-free guidance for conditioned generation.
        
        Performs a forward pass through the policy network with optional classifier-free
        guidance. When cond_scale > 0, runs both conditioned and unconditioned inference
        and combines the outputs using the specified guidance scale.
        
        :param condition: Dictionary with 'cond_scale' and 'mineclip_embeds'
        :param input: Dictionary with 'image' key containing observation images
        :param state_in: Optional list of recurrent state tensors from previous timestep
        :param kwargs: Additional keyword arguments (unused)
        :returns: Tuple of (latents_dict, state_out) where latents_dict contains
                 'pi_logits' and 'vpred' and state_out is updated recurrent state
        """
        condition = input['condition'].copy()
        if 'mineclip_embeds' not in condition:
            condition = self.prepare_condition(condition)
        if state_in is None:
            state_in = self.initial_state(batch_size=input['image'].shape[0], condition=condition)
        input, state_in = input.copy(), state_in.copy()
        mineclip_embeds = condition['mineclip_embeds']
        if mineclip_embeds.shape[0] == 1 and input['image'].shape[0] > 1:
            mineclip_embeds = repeat(mineclip_embeds, '1 ... -> b ...', b=input['image'].shape[0])
        if isinstance(condition['cond_scale'], torch.Tensor):
            condition['cond_scale'] = condition['cond_scale'][0][0].item() #! TEMPORAL FIXME: remove this when we have a better way to handle this
        if condition['cond_scale'] != 0 and condition['cond_scale'] is not None:
            state_in = [rearrange(x, "b ... c -> (b c) ...") for x in state_in]
            images = repeat(input['image'], "b ... -> (b c) ...", c=2)
            mineclip_embeds = rearrange(
                torch.stack([mineclip_embeds, torch.zeros_like(mineclip_embeds)]),
                'c b ... -> (b c) ...'
            )
        else:
            images = input['image']

        dummy_first = torch.zeros((images.shape[0], images.shape[1]), dtype=torch.bool, device=self.device)

        if images.shape[-1] != 3:
            images = rearrange(images, 'b t c h w -> b t h w c')
        if images.dtype == torch.uint8:
            images = images.float()
        elif images.dtype == torch.float32:
            assert images.abs().max() <= 1.0, "float32 image must be in range [-1, 1]"
            images = images * 255.0
        else:
            raise ValueError("image must be either uint8 or float32.")

        (pi_latent, vf_latent), state_out = self.net(
            ob={"img": images, "mineclip_embed": repeat(mineclip_embeds, 'b c -> b t c', t=images.shape[1])}, 
            context={"first": dummy_first}, 
            state_in=state_in
        )

        pi_logits = self.pi_head(pi_latent)
        vpred = self.value_head(vf_latent)
        if condition['cond_scale'] != 0 and condition['cond_scale'] is not None:
            pi_logits = {k: rearrange(v, '(b c) ... -> b c ...', c=2) for k, v in pi_logits.items()}
            vpred = rearrange(vpred, '(b c) ... -> b c ...', c=2)
            state_out = [rearrange(x, '(b c) ... -> b ... c', c=2) for x in state_out]
            
            pi_logits = {k: (1 + condition['cond_scale']) * v[:, 0] - condition['cond_scale'] * v[:, 1] for k, v in pi_logits.items()}
            vpred = vpred[:, 0]

        latents = {
            "pi_logits": pi_logits, 
            "vpred": vpred, 
        }
        return latents, state_out
    
    def initial_state(
        self, 
        batch_size = None,
        condition: Dict[str, Any] = None,
    ) -> List[torch.Tensor]:
        """
        Initialize recurrent state for policy inference.
        
        Creates initial state tensors for the recurrent layers. When classifier-free
        guidance is enabled (cond_scale > 0), duplicates states for both conditioned
        and unconditioned inference branches.
        
        :param condition: Conditioning dictionary with 'cond_scale' key
        :param batch_size: Batch size for state initialization
        :returns: List of initial state tensors
        """
        initial_state = self.net.initial_state(batch_size)
        if condition is None:
            return initial_state
        if condition['cond_scale'] == 0.0 or condition['cond_scale'] is None:
            return initial_state
        else:
            return [torch.stack([x, x], dim=-1) for x in initial_state]
        
    def merge_state(self, states) -> Optional[List[torch.Tensor]]:
        result_states = []
        for i in range(len(states[0])):
            result_states.append(auto_to_torch(torch.cat([state[i] for state in states], 0), device=self.device))
        return result_states

    def split_state(self, states, split_num) -> Optional[List[List[torch.Tensor]]]:
        result_states = [
            [states[j][i:i+1] for j in range(len(states))]
            for i in range(split_num)
        ]
        return result_states

    def reset_parameters(self):
        """
        Reset all trainable parameters in the policy network.
        
        :raises NotImplementedError: This method is not yet implemented
        """
        super().reset_parameters()
        self.net.reset_parameters()
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        """
        Get the device of the policy model.
        
        :returns: torch.device where the model parameters are located
        """
        return next(self.parameters()).device

def load_steve_one_policy(ckpt_path: str) -> SteveOnePolicy:
    """
    Load a pre-trained STEVE-1 policy from a checkpoint path.
    
    Convenience function to load a STEVE-1 policy using the HuggingFace Hub
    model loading mechanism.
    
    :param ckpt_path: Path or HuggingFace model identifier for the checkpoint
    :returns: Loaded SteveOnePolicy instance
    """
    return SteveOnePolicy.from_pretrained(ckpt_path)

if __name__ == '__main__':
    model = SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official").to("cuda")
    model.eval()
    output, memory = model.get_action(
        input={
            'image': torch.zeros(2, 8, 128, 128, 3).to("cuda"), 
            'condition': {
                'cond_scale': 4.0,
                'text': 'mine dirt',
            }
        },
        state_in=None
    )
    print(output.keys())