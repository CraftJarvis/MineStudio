"""V1 VPT network, preserved for weight parity during the v2 migration.

Source: minestudio/models/vpt/body.py (CraftJarvis, project MIT license).
Only the import boundary changed. Migrate components after numerical parity tests.
"""
from copy import deepcopy
from typing import Dict, Optional
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from minestudio.utils.vpt_lib.impala_cnn import ImpalaCNN
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks

class ImgPreprocessing(nn.Module):
    """Normalize incoming images.

    :param img_statistics: remote path to npz file with a mean and std image. If specified
        normalize images using this.
    :param scale_img: If true and img_statistics not specified, scale incoming images by 1/255.
    """

    def __init__(self, img_statistics: Optional[str] = None, scale_img: bool = True):
        """Initialize ImgPreprocessing.

        :param img_statistics: Path to a .npz file containing 'mean' and 'std' for image normalization.
                               If None, normalization will be a simple scaling. Defaults to None.
        :type img_statistics: Optional[str]
        :param scale_img: If True and `img_statistics` is None, scale images by 1/255.0.
                          Defaults to True.
        :type scale_img: bool
        """
        super().__init__()
        self.img_mean = None
        if img_statistics is not None:
            img_statistics = dict(**np.load(img_statistics))
            self.img_mean = nn.Parameter(th.Tensor(img_statistics["mean"]), requires_grad=False)
            self.img_std = nn.Parameter(th.Tensor(img_statistics["std"]), requires_grad=False)
        else:
            self.ob_scale = 255.0 if scale_img else 1.0

    def forward(self, img):
        """Apply image preprocessing.

        Normalizes the input image tensor. If `img_statistics` was provided during
        initialization, it uses the mean and std from the file. Otherwise, it scales
        the image by `1.0 / self.ob_scale`.

        :param img: The input image tensor.
        :type img: torch.Tensor
        :returns: The preprocessed image tensor.
        :rtype: torch.Tensor
        """
        x = img.to(dtype=th.float32)
        if self.img_mean is not None:
            x = (x - self.img_mean) / self.img_std
        else:
            x = x / self.ob_scale
        return x


class ImgObsProcess(nn.Module):
    """ImpalaCNN followed by a linear layer.

    :param cnn_outsize: impala output dimension
    :param output_size: output size of the linear layer.
    :param dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        """Initialize ImgObsProcess.

        :param cnn_outsize: The output size of the ImpalaCNN.
        :type cnn_outsize: int
        :param output_size: The final output size after the linear layer.
        :type output_size: int
        :param dense_init_norm_kwargs: Keyword arguments for the dense FanInInitReLULayer (linear layer).
                                       Defaults to {}.
        :type dense_init_norm_kwargs: Dict
        :param init_norm_kwargs: Keyword arguments for the convolutional FanInInitReLULayers (within ImpalaCNN).
                                 Defaults to {}.
        :type init_norm_kwargs: Dict
        :param kwargs: Additional keyword arguments passed to ImpalaCNN.
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
        """Process the image observation.

        Passes the image through the ImpalaCNN and then a linear layer.

        :param img: The input image tensor.
        :type img: torch.Tensor
        :returns: The processed image features.
        :rtype: torch.Tensor
        """
        return self.linear(self.cnn(img))

class MinecraftPolicy(nn.Module):
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
        recurrence_type="transformer",
        impala_width=1,
        impala_chans=(16, 32, 32),
        obs_processing_width=256, # Unused in this specific constructor
        hidsize=512,
        single_output=False,
        img_shape=None,
        scale_input_img=True,
        only_img_input=False, # Unused in this specific constructor
        init_norm_kwargs={},
        impala_kwargs={},
        input_shape=None,
        active_reward_monitors=None,
        img_statistics=None,
        first_conv_norm=False,
        diff_mlp_embedding=False, # Unused in this specific constructor
        attention_mask_style="clipped_causal",
        attention_heads=8,
        attention_memory_size=2048,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        n_recurrence_layers=1,
        recurrence_is_residual=True,
        timesteps=None,
        use_pre_lstm_ln=True,
        **unused_kwargs,
    ):
        """Initialize the MinecraftPolicy network.

        This network processes image observations, applies a recurrent layer (e.g., Transformer or LSTM),
        and produces latent representations for policy and value functions.

        :param recurrence_type: Type of recurrence to use. Options: "multi_layer_lstm",
                                "multi_layer_bilstm", "multi_masked_lstm", "transformer", "none".
                                Defaults to "transformer".
        :type recurrence_type: str
        :param impala_width: Width multiplier for ImpalaCNN channels. Defaults to 1.
        :type impala_width: int
        :param impala_chans: Base channels for ImpalaCNN layers. Defaults to (16, 32, 32).
        :type impala_chans: Tuple[int, ...]
        :param obs_processing_width: (Currently unused) Intended width for observation processing. Defaults to 256.
        :type obs_processing_width: int
        :param hidsize: Hidden size for various layers, including output latents. Defaults to 512.
        :type hidsize: int
        :param single_output: If True, policy and value functions share the same final latent.
                              Defaults to False.
        :type single_output: bool
        :param img_shape: Shape of the input image (not directly used by ImgObsProcess constructor but passed to it).
                          Defaults to None.
        :type img_shape: Optional[Tuple[int, ...]]
        :param scale_input_img: Whether to scale input images by 1/255.0 if `img_statistics` is not provided.
                                Defaults to True.
        :type scale_input_img: bool
        :param only_img_input: (Currently unused) Flag for using only image input. Defaults to False.
        :type only_img_input: bool
        :param init_norm_kwargs: Kwargs for FanInInitReLULayer normalization. Defaults to {}.
        :type init_norm_kwargs: Dict
        :param impala_kwargs: Additional kwargs for ImpalaCNN. Defaults to {}.
        :type impala_kwargs: Dict
        :param input_shape: (Currently unused by this constructor) Expected input shape. Defaults to None.
        :type input_shape: Optional[Any]
        :param active_reward_monitors: (Currently unused) Configuration for reward monitors. Defaults to None.
        :type active_reward_monitors: Optional[Dict]
        :param img_statistics: Path to image statistics for normalization. Defaults to None.
        :type img_statistics: Optional[str]
        :param first_conv_norm: Whether to apply normalization after the first convolution in ImpalaCNN.
                                Defaults to False.
        :type first_conv_norm: bool
        :param diff_mlp_embedding: (Currently unused) Flag for differential MLP embedding. Defaults to False.
        :type diff_mlp_embedding: bool
        :param attention_mask_style: Style of attention mask for Transformer. Defaults to "clipped_causal".
        :type attention_mask_style: str
        :param attention_heads: Number of attention heads for Transformer. Defaults to 8.
        :type attention_heads: int
        :param attention_memory_size: Memory size for Transformer attention. Defaults to 2048.
        :type attention_memory_size: int
        :param use_pointwise_layer: Whether to use pointwise feed-forward layers in recurrent blocks.
                                    Defaults to True.
        :type use_pointwise_layer: bool
        :param pointwise_ratio: Ratio for pointwise layer hidden dimension. Defaults to 4.
        :type pointwise_ratio: int
        :param pointwise_use_activation: Whether to use activation in pointwise layer. Defaults to False.
        :type pointwise_use_activation: bool
        :param n_recurrence_layers: Number of recurrent layers (e.g., LSTM layers or Transformer blocks).
                                    Defaults to 1.
        :type n_recurrence_layers: int
        :param recurrence_is_residual: Whether to use residual connections in recurrent blocks.
                                       Defaults to True.
        :type recurrence_is_residual: bool
        :param timesteps: Number of timesteps for recurrence (used by ResidualRecurrentBlocks).
                          Defaults to None.
        :type timesteps: Optional[int]
        :param use_pre_lstm_ln: Whether to use LayerNorm before the recurrent layer (if not Transformer).
                                Defaults to True.
        :type use_pre_lstm_ln: bool
        :param unused_kwargs: Catches any other keyword arguments.
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

        self.pre_lstm_ln = nn.LayerNorm(hidsize) if use_pre_lstm_ln else None
        self.diff_obs_process = None

        self.recurrence_type = recurrence_type
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
        self.final_ln = th.nn.LayerNorm(hidsize)

    def output_latent_size(self):
        """Returns the size of the output latent vector.

        :returns: The hidden size, which is the dimension of the output latents.
        :rtype: int
        """
        return self.hidsize

    def forward(self, ob, state_in, context):
        """Forward pass of the MinecraftPolicy.

        Processes image observations, passes them through recurrent layers, and produces
        latent representations.

        :param ob: Dictionary of observations, expected to contain "image".
        :type ob: Dict[str, torch.Tensor]
        :param state_in: Input recurrent state.
        :type state_in: Any # Type depends on recurrence_type
        :param context: Context dictionary, expected to contain "first" (a tensor indicating episode starts).
        :type context: Dict[str, torch.Tensor]
        :returns: A tuple containing:
            - pi_latent_or_tuple (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                If `single_output` is True, this is a single tensor for both policy and value.
                Otherwise, it's a tuple (pi_latent, vf_latent).
            - state_out (Any): Output recurrent state.
        :rtype: Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Any]
        """
        first = context["first"]
        x = self.img_preprocess(ob["image"])
        x = self.img_process(x)

        if self.diff_obs_process:
            processed_obs = self.diff_obs_process(ob["diff_goal"])
            x = processed_obs + x

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
        """Get the initial recurrent state.

        :param batchsize: The batch size for the initial state.
        :type batchsize: int
        :returns: The initial recurrent state, or None if no recurrent layer is used.
        :rtype: Any # Type depends on recurrence_type, can be None
        """
        if self.recurrent_layer:
            return self.recurrent_layer.initial_state(batchsize)
        else:
            return None
