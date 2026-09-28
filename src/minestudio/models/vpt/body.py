'''
Date: 2024-11-11 20:54:15
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-05-28 14:29:56
FilePath: /MineStudio/minestudio/models/vpt/body.py
'''
import os
import pickle
import gymnasium
import numpy as np
import torch
import torch as th
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Optional, Callable, Union, Tuple, Any

from huggingface_hub import PyTorchModelHubMixin
from minestudio.utils.vpt_lib.impala_cnn import ImpalaCNN
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from minestudio.models.base_policy import MinePolicy
from minestudio.online.utils import auto_stack, auto_to_torch
from minestudio.utils.register import Registers

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

@Registers.model.register
class VPTPolicy(MinePolicy, PyTorchModelHubMixin):
    """VPT (Video PreTraining) Policy.

    This class wraps the `MinecraftPolicy` network and integrates it with the `MinePolicy`
    base class, providing methods for action selection, initial state, and state/input
    merging/splitting for batched online inference. It also supports loading from
    Hugging Face Hub.

    :param policy_kwargs: Keyword arguments to initialize the `MinecraftPolicy` (self.net).
    :type policy_kwargs: Dict
    :param action_space: The action space of the environment. Passed to `MinePolicy` constructor.
                         Defaults to None.
    :type action_space: Optional[gymnasium.spaces.Space]
    :param kwargs: Additional keyword arguments passed to the `MinePolicy` constructor (e.g., temperature).
    """

    def __init__(self, policy_kwargs, action_space=None, **kwargs):
        """Initialize VPTPolicy.

        :param policy_kwargs: Keyword arguments for the underlying `MinecraftPolicy`.
        :type policy_kwargs: Dict
        :param action_space: Action space for the policy.
        :type action_space: Optional[gymnasium.spaces.Space]
        :param kwargs: Additional keyword arguments for `MinePolicy` (e.g., temperature, nucleus_prob).
        """
        super().__init__(hiddim=policy_kwargs["hidsize"], action_space=action_space, **kwargs)
        self.net = MinecraftPolicy(**policy_kwargs)
        self.cached_init_states = dict()

    def initial_state(self, batch_size: int=None):
        """Get the initial recurrent state for a given batch size.

        Caches initial states for frequently used batch sizes.

        :param batch_size: The batch size. If None, returns state for batch size 1 (squeezed).
                           Defaults to None.
        :type batch_size: Optional[int]
        :returns: A list of initial state tensors for the recurrent network, moved to the correct device.
        :rtype: List[torch.Tensor]
        """
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.net.initial_state(1)]
        else:
            if batch_size not in self.cached_init_states:
                self.cached_init_states[batch_size] = [t.to(self.device) for t in self.net.initial_state(batch_size)]
            return self.cached_init_states[batch_size]

    def forward(self, input, state_in, **kwargs):
        """Forward pass of the VPTPolicy.

        Takes observations and recurrent state, passes them through the underlying
        `MinecraftPolicy` network, and then through policy and value heads.

        :param input: Dictionary of input observations, expected to contain "image".
                      The "image" tensor should have shape (B, T, H, W, C) or similar.
        :type input: Dict[str, torch.Tensor]
        :param state_in: Input recurrent state. If None, an initial state is generated.
        :type state_in: Optional[List[torch.Tensor]]
        :param kwargs: Additional keyword arguments (not directly used in this method but part of signature).
        :returns: A tuple containing:
            - latents (Dict[str, torch.Tensor]): Dictionary with 'pi_logits' and 'vpred'.
            - state_out (List[torch.Tensor]): Output recurrent state.
        :rtype: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]
        """
        B, T = input["image"].shape[:2]
        first = torch.tensor([[False]], device=self.device).repeat(B, T)
        state_in = self.initial_state(B) if state_in is None else state_in

        #input: 1, 128, 128, 128, 3
        #first: 1, 128
        # state_in[0]: 1, 1, 1, 128
        # state_in[1]: 1, 1, 128, 128
        try:
            (pi_h, v_h), state_out = self.net(input, state_in, context={"first": first})
        except Exception as e:
            import ray
            ray.util.pdb.set_trace()
        pi_logits = self.pi_head(pi_h)
        vpred = self.value_head(v_h)
        latents = {'pi_logits': pi_logits, 'vpred': vpred}
        return latents, state_out
    
    def merge_input(self, inputs) -> torch.tensor:
        """Merge a list of individual inputs into a single batched input tensor.

        Handles inputs where "image" is 3D (single frame) or 4D (already batched/sequence).
        All inputs are moved to the policy's device.

        :param inputs: A list of input dictionaries, each expected to have an "image" key.
        :type inputs: List[Dict[str, Any]] # Values are typically np.ndarray or torch.Tensor
        :returns: A batched input dictionary with "image" as a torch.Tensor.
        :rtype: Dict[str, torch.Tensor]
        """
        inputs = auto_to_torch(inputs, device=self.device)
        if inputs[0]["image"].dim() == 3:
            in_inputs=[{"image": input["image"]} for input in inputs]
            out_inputs = auto_to_torch(auto_stack([auto_stack([input]) for input in in_inputs]), device=self.device)
            return out_inputs
        elif inputs[0]["image"].dim() == 4:
            out_inputs = auto_to_torch(auto_stack([input["image"] for input in inputs]), device=self.device)
            return out_inputs
        
    def merge_state(self, states) -> Optional[List[torch.Tensor]]:
        """Merge a list of individual recurrent states into a single batched state.

        Concatenates corresponding state tensors along the batch dimension.

        :param states: A list of recurrent states. Each state is a list of tensors.
                       Example: `[[s1_env1, s2_env1, ...], [s1_env2, s2_env2, ...], ...]`
        :type states: List[List[torch.Tensor]]
        :returns: The batched recurrent state, where each tensor is a concatenation of
                  the corresponding tensors from the input states.
        :rtype: Optional[List[torch.Tensor]]
        """
        result_states = []
        for i in range(len(states[0])):
            result_states.append(auto_to_torch(torch.cat([state[i] for state in states], 0), device=self.device))
        return result_states

    def split_state(self, states, split_num) -> Optional[List[List[torch.Tensor]]]:
        """Split a batched recurrent state into a list of individual states.

        :param states: The batched recurrent state (a list of tensors, where the first
                       dimension of each tensor is the batch size).
        :type states: List[torch.Tensor]
        :param split_num: The number of individual states to split into (should match batch size).
        :type split_num: int
        :returns: A list of individual recurrent states. Each state in the list corresponds
                  to one item from the original batch.
                  Example: `[[s1_item1, s2_item1, ...], [s1_item2, s2_item2, ...], ...]`
        :rtype: Optional[List[List[torch.Tensor]]]
        """
        result_states = [
            [states[j][i:i+1] for j in range(len(states))]
            for i in range(split_num)
        ]
        return result_states

@Registers.model_loader.register
def load_vpt_policy(model_path: str, weights_path: Optional[str] = None):
    """Load a VPTPolicy model.

    Can load from a local pickle file and optionally apply weights from a .ckpt file,
    or load a pretrained model from Hugging Face Hub if `model_path` is None.

    :param model_path: Path to the .model pickle file containing policy configuration.
                       If None, attempts to load from Hugging Face Hub.
    :type model_path: Optional[str]
    :param weights_path: Path to the .ckpt file containing model weights.
                         If None, weights are not loaded separately (e.g., if part of .model or Hub model).
                         Defaults to None.
    :type weights_path: Optional[str]
    :raises ValueError: If `model_path` is None and `ckpt_path` (internal, seems like a typo for `weights_path`
                        in the original conditional logic, but `repo_id` is used if `weights_path` is also None)
                        is also None, and no default repo_id is hit.
    :returns: The loaded VPTPolicy model.
    :rtype: VPTPolicy
    """
    if model_path is None:
        # The original code had a variable `ckpt_path` here which is not defined.
        # Assuming the intent was to use `weights_path` or a default repo_id.
        repo_id = "CraftJarvis/MineStudio_VPT.rl_for_shoot_animals_2x" # Default if no other info
        # if weights_path is None: # This condition was how repo_id was set in original
        #     repo_id = "CraftJarvis/MineStudio_VPT.rl_for_shoot_animals_2x"
        return VPTPolicy.from_pretrained(f"{repo_id}")

    model = pickle.load(Path(model_path).open("rb"))
    policy_kwargs = model['model']['args']['net']['args']
    vpt_policy = VPTPolicy(
        policy_kwargs=model['model']['args']['net']['args'], 
        temperature=model['model']['args']['pi_head_opts']['temperature']
    )
    if weights_path is None:
        return vpt_policy
    weights = torch.load(weights_path, map_location='cpu')
    if 'state_dict' in weights:
        weights = {k.replace('mine_policy.', ''): v for k, v in weights['state_dict'].items()}
    weights = {k: v for k, v in weights.items() if k in vpt_policy.state_dict()}
    vpt_policy.load_state_dict(weights, strict=True)
    return vpt_policy

if __name__ == '__main__':
    # model = load_vpt_policy(
    #     model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
    #     weights_path="/nfs-shared-2/hekaichen/minestudio_checkpoint/gate.ckpt"
    # ).to("cuda")
    # model.push_to_hub("CraftJarvis/MineStudio_VPT.rl_for_build_portal_2x")
    model = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_for_shoot_animals_2x").to("cuda")
    model.eval()
    dummy_input = {
        "image": torch.zeros(1, 1, 128, 128, 3).to("cuda"),
    }
    output, memory = model(dummy_input, None)
    print(output)