from typing import Dict, Optional
import rich
from rich.console import Console
import torch
import torch as th
from torch import nn
from torch.nn import functional as F

import minestudio.utils.vpt_lib.torch_util as tu
from minestudio.utils.vpt_lib.masked_attention import MaskedAttention
from minestudio.utils.vpt_lib.minecraft_util import store_args
from minestudio.utils.vpt_lib.tree_util import tree_map

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class FiLMBlock(nn.Module):
    
    def __init__(self, hidsize: int):
        super().__init__()
        self.fc = nn.Linear(hidsize, 2*hidsize)
        self.ff = FeedForward(hidsize, mult = 2)
        self.norm = nn.LayerNorm(hidsize)
    
    def forward(self, x, z):
        z = self.ff(z) + z
        gamma, beta = self.fc(z).chunk(2, dim=-1)
        y = x * (1 + gamma) + beta
        y = self.norm(y) #! LayerNorm after FiLM
        return y


class AdaLNBlock(nn.Module):
    
    def __init__(self, hidsize, **kwargs) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidsize, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(hidsize, hidsize*2, bias=True)
        )
    
    def forward(self, x, z):
        assert z.shape[1] == 1 and z.shape[-1] == x.shape[-1]
        shift, scale = self.adaLN_modulation(z).chunk(2, dim=-1)
        x = self.norm(x) * scale + shift
        return x

def get_module_log_keys_recursive(m: nn.Module):
    """Recursively get all keys that a module and its children want to log."""
    keys = []
    if hasattr(m, "get_log_keys"):
        keys += m.get_log_keys()
    for c in m.children():
        keys += get_module_log_keys_recursive(c)
    return keys


class FanInInitReLULayer(nn.Module):
    """Implements a slightly modified init that correctly produces std 1 outputs given ReLU activation
    :param inchan: number of input channels
    :param outchan: number of output channels
    :param layer_args: positional layer args
    :param layer_type: options are "linear" (dense layer), "conv" (2D Convolution), "conv3d" (3D convolution)
    :param init_scale: multiplier on initial weights
    :param batch_norm: use batch norm after the layer (for 2D data)
    :param group_norm_groups: if not None, use group norm with this many groups after the layer. Group norm 1
        would be equivalent of layernorm for 2D data.
    :param layer_norm: use layernorm after the layer (for 1D data)
    :param layer_kwargs: keyword arguments for the layer
    """

    @store_args
    def __init__(
        self,
        inchan: int,
        outchan: int,
        *layer_args,
        layer_type: str = "conv",
        init_scale: int = 1,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation=True,
        log_scope: Optional[str] = None,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
        self.layer = layer(inchan, outchan, bias=self.norm is None, *layer_args, **layer_kwargs)

        # Init Weights (Fan-In)
        self.layer.weight.data *= init_scale / self.layer.weight.norm(
            dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        )
        # Init Bias
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

    def forward(self, x):
        """Norm after the activation. Experimented with this for both IAM and BC and it was slightly better."""
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x

    def get_log_keys(self):
        return [
            f"activation_mean/{self.log_scope}",
            f"activation_std/{self.log_scope}",
        ]


class ResidualRecurrentBlocks(nn.Module):
    @store_args
    def __init__(
        self,
        n_block=2,
        recurrence_type="multi_layer_lstm",
        is_residual=True,
        word_dropout=0.0,
        **block_kwargs,
    ):
        super().__init__()
        init_scale = n_block ** -0.5 if is_residual else 1
        self.blocks = nn.ModuleList(
            [
                ResidualRecurrentBlock(
                    **block_kwargs,
                    recurrence_type=recurrence_type,
                    is_residual=is_residual,
                    init_scale=init_scale,
                    block_number=i,
                )
                for i in range(n_block)
            ]
        )
        self.word_dropout = word_dropout
        if word_dropout > 0.0:
            Console().log(f"Enable word dropout = {word_dropout} in ResidualRecurrentBlock.")


    def forward(self, x, first, state, **kwargs):
        
        if self.word_dropout > 0. and self.training:
            b, t, T = x.shape[0], x.shape[1], state[0].shape[-1]+x.shape[1]
            kwargs['word_mask'] = self.get_word_mask(b, t, T, x.device)
        
        state_out = []
        if 'lstm' in self.recurrence_type:
            assert len(state) == len(
                self.blocks
            ), f"Length of state {len(state)} did not match length of blocks {len(self.blocks)}"
        else:
            # FIXME: transformer only, len(state) == 3 * self.blocks state[0].shape
            # [N]->[N, 1, 128]; [N, 128, 1024]; [N, 128, 1024]
            assert len(state) == 3 * len(
                self.blocks
            ), f"Length of state {len(state)} did not match length of blocks {len(self.blocks)}"
            n_state = []
            for i in range(len(self.blocks)):
                n_state.append((state[3*i], (state[3*i+1], state[3*i+2])))
            state = n_state
        
        for block, _s_in in zip(self.blocks, state):
            x, _s_o = block(x, first, _s_in, **kwargs)
            state_out.append(_s_o)

        # FIXME: transformer only, len(state) == 3 * self.blocks
        if 'lstm' not in self.recurrence_type:
            n_state_out = []
            for i in range(len(state_out)):
                n_state_out.append(state_out[i][0])
                n_state_out.append(state_out[i][1][0])
                n_state_out.append(state_out[i][1][1])
            state_out = n_state_out

        return x, state_out

    def get_word_mask(self, b, t, T, device: th.device):
        return (th.rand((b, t, T), device=device) > self.word_dropout)

    def initial_state(self, batchsize):
        if "lstm" in self.recurrence_type:
            return [None for b in self.blocks]
        else:
            # FIXME: transformer only, len(state) == 3 * self.blocks
            ret = [b.r.initial_state(batchsize) for b in self.blocks]
            n_ret = []
            for i in range(len(ret)):
                n_ret.append(ret[i][0])
                n_ret.append(ret[i][1][0])
                n_ret.append(ret[i][1][1])
            return n_ret


class ResidualRecurrentBlock(nn.Module):
    @store_args
    def __init__(
        self,
        hidsize,
        timesteps,
        init_scale=1,
        recurrence_type="multi_layer_lstm",
        is_residual=True,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        attention_heads=8,
        attention_memory_size=2048,
        attention_mask_style="clipped_causal",
        log_scope="resblock",
        block_number=0,
        inject_condition=False,
    ):
        super().__init__()
        self.log_scope = f"{log_scope}{block_number}"
        s = init_scale
        if use_pointwise_layer:
            if is_residual:
                s *= 2 ** -0.5  # second residual
            self.mlp0 = FanInInitReLULayer(
                hidsize,
                hidsize * pointwise_ratio,
                init_scale=1,
                layer_type="linear",
                layer_norm=True,
                log_scope=self.log_scope + "/ptwise_mlp0",
            )
            self.mlp1 = FanInInitReLULayer(
                hidsize * pointwise_ratio,
                hidsize,
                init_scale=s,
                layer_type="linear",
                use_activation=pointwise_use_activation,
                log_scope=self.log_scope + "/ptwise_mlp1",
            )

        self.pre_r_ln = nn.LayerNorm(hidsize)
        if recurrence_type in ["multi_layer_lstm", "multi_layer_bilstm"]:
            self.r = nn.LSTM(hidsize, hidsize, batch_first=True)
            nn.init.normal_(self.r.weight_hh_l0, std=s * (self.r.weight_hh_l0.shape[0] ** -0.5))
            nn.init.normal_(self.r.weight_ih_l0, std=s * (self.r.weight_ih_l0.shape[0] ** -0.5))
            self.r.bias_hh_l0.data *= 0
            self.r.bias_ih_l0.data *= 0
        elif recurrence_type == "transformer":
            self.r = MaskedAttention(
                input_size=hidsize,
                timesteps=timesteps,
                memory_size=attention_memory_size,
                heads=attention_heads,
                init_scale=s,
                norm="none",
                log_scope=log_scope + "/sa",
                use_muP_factor=True,
                mask=attention_mask_style,
            )
        
        self.inject_condition = inject_condition
        if self.inject_condition:
            self.injection = FiLMBlock(hidsize=hidsize)
            # self.injection = GatedCrossAttentionBlock(hidsize=hidsize, heads=attention_heads)
            # self.injection = AdaLNBlock(hidsize=hidsize)


    def forward(self, x, first, state, **kwargs):
        
        residual = x
        
        x = self.pre_r_ln(x)
        if self.inject_condition:
            x = self.injection(x=x, z=kwargs['ce_latent'])
        
        x, state_out = recurrent_forward(
            self.r,
            x,
            first,
            state,
            reverse_lstm=self.recurrence_type == "multi_layer_bilstm" and (self.block_number + 1) % 2 == 0,
            **kwargs,
        )
        if self.is_residual and "lstm" in self.recurrence_type:  # Transformer already residual.
            x = x + residual
        if self.use_pointwise_layer:
            # Residual MLP
            residual = x
            x = self.mlp1(self.mlp0(x))
            if self.is_residual:
                x = x + residual
        return x, state_out


def recurrent_forward(module, x, first, state, reverse_lstm=False, **kwargs):
    if isinstance(module, nn.LSTM):
        if state is not None:
            mask = 1 - first[:, 0, None, None].to(th.float)
            state = tree_map(lambda _s: _s * mask, state)
            state = tree_map(lambda _s: _s.transpose(0, 1), state)  # NL, B, H
        if reverse_lstm:
            x = th.flip(x, [1])
        x, state_out = module(x, state)
        if reverse_lstm:
            x = th.flip(x, [1])
        state_out = tree_map(lambda _s: _s.transpose(0, 1), state_out)  # B, NL, H
        return x, state_out
    else:
        return module(x, first, state, **kwargs)


def _banded_repeat(x, t):
    """
    Repeats x with a shift.
    For example (ignoring the batch dimension):

    _banded_repeat([A B C D E], 4)
    =
    [D E 0 0 0]
    [C D E 0 0]
    [B C D E 0]
    [A B C D E]
    """
    b, T = x.shape
    padding_x = (0, t - 1)
    x = F.pad(x, padding_x, "constant", 0)
    result = x.unfold(1, T, 1).flip(1)
    return result


def bandify(b_nd, t, T):
    """
    b_nd -> D_ntT, where
        "n" indexes over basis functions
        "d" indexes over time differences
        "t" indexes over output time
        "T" indexes over input time
        only t >= T is nonzero
    B_ntT[n, t, T] = b_nd[n, t - T]
    """
    nbasis, bandsize = b_nd.shape
    b_nd = torch.flip(b_nd, [1])
    if bandsize >= T:
        b_nT = b_nd[:, -T:]
    else:
        padding_corrected = (T - bandsize, 0)
        b_nT = F.pad(b_nd, padding_corrected, "constant", 0)
    D_tnT = _banded_repeat(b_nT, t)
    return D_tnT


def get_norm(name, d, dtype=th.float32):
    if name == "none":
        return lambda x: x
    elif name == "layer":
        return tu.LayerNorm(d, dtype=dtype)
    else:
        raise NotImplementedError(name)

# from typing import Dict, Optional

# import torch as th
# from torch import nn
# from torch.nn import functional as F

# import src.utils.vpt_lib.torch_util as tu
# from src.utils.vpt_lib.masked_attention import MaskedAttention
# from src.utils.vpt_lib.minecraft_util import store_args
# from src.utils.vpt_lib.tree_util import tree_map


# def get_module_log_keys_recursive(m: nn.Module):
#     """Recursively get all keys that a module and its children want to log."""
#     keys = []
#     if hasattr(m, "get_log_keys"):
#         keys += m.get_log_keys()
#     for c in m.children():
#         keys += get_module_log_keys_recursive(c)
#     return keys


# class FanInInitReLULayer(nn.Module):
#     """Implements a slightly modified init that correctly produces std 1 outputs given ReLU activation
#     :param inchan: number of input channels
#     :param outchan: number of output channels
#     :param layer_args: positional layer args
#     :param layer_type: options are "linear" (dense layer), "conv" (2D Convolution), "conv3d" (3D convolution)
#     :param init_scale: multiplier on initial weights
#     :param batch_norm: use batch norm after the layer (for 2D data)
#     :param group_norm_groups: if not None, use group norm with this many groups after the layer. Group norm 1
#         would be equivalent of layernorm for 2D data.
#     :param layer_norm: use layernorm after the layer (for 1D data)
#     :param layer_kwargs: keyword arguments for the layer
#     """

#     @store_args
#     def __init__(
#         self,
#         inchan: int,
#         outchan: int,
#         *layer_args,
#         layer_type: str = "conv",
#         init_scale: int = 1,
#         batch_norm: bool = False,
#         batch_norm_kwargs: Dict = {},
#         group_norm_groups: Optional[int] = None,
#         layer_norm: bool = False,
#         use_activation=True,
#         log_scope: Optional[str] = None,
#         **layer_kwargs,
#     ):
#         super().__init__()

#         # Normalization
#         self.norm = None
#         if batch_norm:
#             self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
#         elif group_norm_groups is not None:
#             self.norm = nn.GroupNorm(group_norm_groups, inchan)
#         elif layer_norm:
#             self.norm = nn.LayerNorm(inchan)

#         layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
#         self.layer = layer(inchan, outchan, bias=self.norm is None, *layer_args, **layer_kwargs)

#         # Init Weights (Fan-In)
#         self.layer.weight.data *= init_scale / self.layer.weight.norm(
#             dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
#         )
#         # Init Bias
#         if self.layer.bias is not None:
#             self.layer.bias.data *= 0

#     def forward(self, x):
#         """Norm after the activation. Experimented with this for both IAM and BC and it was slightly better."""
#         if self.norm is not None:
#             x = self.norm(x)
#         x = self.layer(x)
#         if self.use_activation:
#             x = F.relu(x, inplace=True)
#         return x

#     def get_log_keys(self):
#         return [
#             f"activation_mean/{self.log_scope}",
#             f"activation_std/{self.log_scope}",
#         ]


# class ResidualRecurrentBlocks(nn.Module):
#     @store_args
#     def __init__(
#         self,
#         n_block=2,
#         recurrence_type="multi_layer_lstm",
#         is_residual=True,
#         **block_kwargs,
#     ):
#         super().__init__()
#         init_scale = n_block ** -0.5 if is_residual else 1
#         self.blocks = nn.ModuleList(
#             [
#                 ResidualRecurrentBlock(
#                     **block_kwargs,
#                     recurrence_type=recurrence_type,
#                     is_residual=is_residual,
#                     init_scale=init_scale,
#                     block_number=i,
#                 )
#                 for i in range(n_block)
#             ]
#         )

#     def forward(self, x, first, state):
#         state_out = []
#         assert len(state) == len(
#             self.blocks
#         ), f"Length of state {len(state)} did not match length of blocks {len(self.blocks)}"
#         for block, _s_in in zip(self.blocks, state):
#             x, _s_o = block(x, first, _s_in)
#             state_out.append(_s_o)
#         return x, state_out

#     def initial_state(self, batchsize):
#         if "lstm" in self.recurrence_type:
#             return [None for b in self.blocks]
#         else:
#             return [b.r.initial_state(batchsize) for b in self.blocks]


# class ResidualRecurrentBlock(nn.Module):
#     @store_args
#     def __init__(
#         self,
#         hidsize,
#         timesteps,
#         init_scale=1,
#         recurrence_type="multi_layer_lstm",
#         is_residual=True,
#         use_pointwise_layer=True,
#         pointwise_ratio=4,
#         pointwise_use_activation=False,
#         attention_heads=8,
#         attention_memory_size=2048,
#         attention_mask_style="clipped_causal",
#         log_scope="resblock",
#         block_number=0,
#     ):
#         super().__init__()
#         self.log_scope = f"{log_scope}{block_number}"
#         s = init_scale
#         if use_pointwise_layer:
#             if is_residual:
#                 s *= 2 ** -0.5  # second residual
#             self.mlp0 = FanInInitReLULayer(
#                 hidsize,
#                 hidsize * pointwise_ratio,
#                 init_scale=1,
#                 layer_type="linear",
#                 layer_norm=True,
#                 log_scope=self.log_scope + "/ptwise_mlp0",
#             )
#             self.mlp1 = FanInInitReLULayer(
#                 hidsize * pointwise_ratio,
#                 hidsize,
#                 init_scale=s,
#                 layer_type="linear",
#                 use_activation=pointwise_use_activation,
#                 log_scope=self.log_scope + "/ptwise_mlp1",
#             )

#         self.pre_r_ln = nn.LayerNorm(hidsize)
#         if recurrence_type in ["multi_layer_lstm", "multi_layer_bilstm"]:
#             self.r = nn.LSTM(hidsize, hidsize, batch_first=True)
#             nn.init.normal_(self.r.weight_hh_l0, std=s * (self.r.weight_hh_l0.shape[0] ** -0.5))
#             nn.init.normal_(self.r.weight_ih_l0, std=s * (self.r.weight_ih_l0.shape[0] ** -0.5))
#             self.r.bias_hh_l0.data *= 0
#             self.r.bias_ih_l0.data *= 0
#         elif recurrence_type == "transformer":
#             self.r = MaskedAttention(
#                 input_size=hidsize,
#                 timesteps=timesteps,
#                 memory_size=attention_memory_size,
#                 heads=attention_heads,
#                 init_scale=s,
#                 norm="none",
#                 log_scope=log_scope + "/sa",
#                 use_muP_factor=True,
#                 mask=attention_mask_style,
#             )

#     def forward(self, x, first, state):
#         residual = x
#         x = self.pre_r_ln(x)
#         x, state_out = recurrent_forward(
#             self.r,
#             x,
#             first,
#             state,
#             reverse_lstm=self.recurrence_type == "multi_layer_bilstm" and (self.block_number + 1) % 2 == 0,
#         )
#         if self.is_residual and "lstm" in self.recurrence_type:  # Transformer already residual.
#             x = x + residual
#         if self.use_pointwise_layer:
#             # Residual MLP
#             residual = x
#             x = self.mlp1(self.mlp0(x))
#             if self.is_residual:
#                 x = x + residual
#         return x, state_out


# def recurrent_forward(module, x, first, state, reverse_lstm=False):
#     if isinstance(module, nn.LSTM):
#         if state is not None:
#             # In case recurrent models do not accept a "first" argument we zero out the hidden state here
#             mask = 1 - first[:, 0, None, None].to(th.float)
#             state = tree_map(lambda _s: _s * mask, state)
#             state = tree_map(lambda _s: _s.transpose(0, 1), state)  # NL, B, H
#         if reverse_lstm:
#             x = th.flip(x, [1])
#         x, state_out = module(x, state)
#         if reverse_lstm:
#             x = th.flip(x, [1])
#         state_out = tree_map(lambda _s: _s.transpose(0, 1), state_out)  # B, NL, H
#         return x, state_out
#     else:
#         return module(x, first, state)


# def _banded_repeat(x, t):
#     """
#     Repeats x with a shift.
#     For example (ignoring the batch dimension):

#     _banded_repeat([A B C D E], 4)
#     =
#     [D E 0 0 0]
#     [C D E 0 0]
#     [B C D E 0]
#     [A B C D E]
#     """
#     b, T = x.shape
#     x = th.cat([x, x.new_zeros(b, t - 1)], dim=1)
#     result = x.unfold(1, T, 1).flip(1)
#     return result


# def bandify(b_nd, t, T):
#     """
#     b_nd -> D_ntT, where
#         "n" indexes over basis functions
#         "d" indexes over time differences
#         "t" indexes over output time
#         "T" indexes over input time
#         only t >= T is nonzero
#     B_ntT[n, t, T] = b_nd[n, t - T]
#     """
#     nbasis, bandsize = b_nd.shape
#     b_nd = b_nd[:, th.arange(bandsize - 1, -1, -1)]
#     if bandsize >= T:
#         b_nT = b_nd[:, -T:]
#     else:
#         b_nT = th.cat([b_nd.new_zeros(nbasis, T - bandsize), b_nd], dim=1)
#     D_tnT = _banded_repeat(b_nT, t)
#     return D_tnT


# def get_norm(name, d, dtype=th.float32):
#     if name == "none":
#         return lambda x: x
#     elif name == "layer":
#         return tu.LayerNorm(d, dtype=dtype)
#     else:
#         raise NotImplementedError(name)
