'''
Date: 2024-11-11 15:59:37
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-05-26 21:41:12
FilePath: /MineStudio/minestudio/models/base_policy.py
'''
from abc import ABC, abstractmethod
import numpy as np
import torch
import typing
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
import gymnasium
from einops import rearrange
from minestudio.utils.vpt_lib.action_head import make_action_head
from minestudio.utils.vpt_lib.normalize_ewma import NormalizeEwma
from minestudio.utils.vpt_lib.scaled_mse_head import ScaledMSEHead

def dict_map(fn, d):
    """Recursively apply a function to all values in a dictionary or DictConfig.

    :param fn: The function to apply.
    :param d: The dictionary or DictConfig.
    :returns: A new dictionary with the function applied to its values.
    """
    if isinstance(d, Dict) or isinstance(d, DictConfig):
        return {k: dict_map(fn, v) for k, v in d.items()}
    else:
        return fn(d)

T = typing.TypeVar("T")
def recursive_tensor_op(fn, d: T) -> T:
    """Recursively apply a function to all tensors in a nested structure of lists, tuples, or dictionaries.

    :param fn: The function to apply to tensors.
    :param d: The nested structure containing tensors.
    :type d: typing.TypeVar("T")
    :returns: A new nested structure with the function applied to its tensors.
    :rtype: typing.TypeVar("T")
    :raises ValueError: if an unexpected type is encountered.
    """
    if isinstance(d, torch.Tensor):
        return fn(d)
    elif isinstance(d, list):
        return [recursive_tensor_op(fn, elem) for elem in d] # type: ignore
    elif isinstance(d, tuple):
        return tuple(recursive_tensor_op(fn, elem) for elem in d) # type: ignore
    elif isinstance(d, dict):
        return {k: recursive_tensor_op(fn, v) for k, v in d.items()} # type: ignore
    elif d is None:
        return None # type: ignore
    else:
        raise ValueError(f"Unexpected type {type(d)}")

class MinePolicy(torch.nn.Module, ABC):
    """Abstract base class for Minecraft policies.

    This class defines the basic interface for a policy, including methods for
    getting actions, computing initial states, and resetting parameters. It also
    handles batching and unbatching of inputs and states.

    :param hiddim: The hidden dimension size.
    :param action_space: The action space of the environment. Defaults to a predefined
                         Dict space for "camera" and "buttons" if None.
    :param temperature: Temperature parameter for sampling actions from the policy head.
                        Defaults to 1.0.
    :param nucleus_prob: Nucleus probability for sampling actions. Defaults to None.
    """
    def __init__(self, hiddim, action_space=None, temperature=1.0, nucleus_prob=None) -> None:
        torch.nn.Module.__init__(self)
        if action_space is None:
            action_space = gymnasium.spaces.Dict({
                "camera": gymnasium.spaces.MultiDiscrete([121]), 
                "buttons": gymnasium.spaces.MultiDiscrete([8641]),
            })
        self.value_head = ScaledMSEHead(hiddim, 1, norm_type="ewma", norm_kwargs=None)
        self.pi_head = make_action_head(action_space, hiddim, temperature=temperature, nucleus_prob=nucleus_prob)

    def reset_parameters(self):
        """Resets the parameters of the policy and value heads."""
        self.pi_head.reset_parameters()
        self.value_head.reset_parameters()

    @abstractmethod
    def forward(self,
                input: Dict[str, Any],
                state_in: Optional[List[torch.Tensor]] = None,
                **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """Abstract method for the forward pass of the policy.

        Subclasses must implement this method to define the policy's computation.

        :param input: A dictionary of input tensors.
        :type input: Dict[str, Any]
        :param state_in: An optional list of input state tensors.
        :type state_in: Optional[List[torch.Tensor]]
        :param kwargs: Additional keyword arguments.
        :returns: A tuple containing:
            - latents (Dict[str, torch.Tensor]): A dictionary containing `pi_logits` and `vpred` latent tensors.
            - state_out (List[torch.Tensor]): A list containing the updated state tensors.
        :rtype: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]
        """
        pass

    @abstractmethod
    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        """Abstract method to get the initial state of the policy.

        Subclasses must implement this method.

        :param batch_size: The batch size for the initial state. Defaults to None.
        :type batch_size: Optional[int]
        :returns: A list of initial state tensors.
        :rtype: List[torch.Tensor]
        """
        pass

    @torch.inference_mode()
    def get_action(self,
                   input: Dict[str, Any],
                   state_in: Optional[List[torch.Tensor]],
                   deterministic: bool = False,
                   input_shape: str = "BT*",
                   **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """Gets an action from the policy.

        This method performs a forward pass, samples an action, and handles
        different input shapes.

        :param input: A dictionary of input tensors.
        :type input: Dict[str, Any]
        :param state_in: An optional list of input state tensors.
        :type state_in: Optional[List[torch.Tensor]]
        :param deterministic: Whether to sample actions deterministically. Defaults to False.
        :type deterministic: bool
        :param input_shape: The shape of the input. Can be "*" (single instance) or "BT*"
                            (batched sequence). Defaults to "BT*".
        :type input_shape: str
        :param kwargs: Additional keyword arguments.
        :returns: A tuple containing:
            - action (Dict[str, torch.Tensor]): The sampled action.
            - state_out (List[torch.Tensor]): The updated state.
        :rtype: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]
        :raises NotImplementedError: if `input_shape` is not "*" or "BT*".
        """
        if input_shape == "*":
            input = dict_map(self._batchify, input)
            if state_in is not None:
                state_in = recursive_tensor_op(lambda x: x.unsqueeze(0), state_in)
        elif input_shape != "BT*":
            raise NotImplementedError
        latents, state_out = self.forward(input, state_in, **kwargs)
        action = self.pi_head.sample(latents['pi_logits'], deterministic)
        self.vpred = latents['vpred']
        if input_shape == "BT*":
            self.cache_latents = latents
            return action, state_out
        elif input_shape == "*":
            self.cache_latents = dict_map(lambda tensor: tensor[0][0], latents)
            return dict_map(lambda tensor: tensor[0][0], action), recursive_tensor_op(lambda x: x[0], state_out)
        else:
            raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Gets the device of the policy's parameters.

        :returns: The device (e.g., 'cpu', 'cuda').
        :rtype: torch.device
        """
        return next(self.parameters()).device

    def _batchify(self, elem):
        """Converts a single data element to a batched tensor on the correct device.

        Handles integers, floats, numpy arrays, and PyTorch tensors. Strings are
        wrapped in a nested list.

        :param elem: The element to batchify.
        :returns: The batchified element as a tensor or nested list for strings.
        """
        if isinstance(elem, (int, float)):
            elem = torch.tensor(elem, device=self.device)
        if isinstance(elem, np.ndarray):
            return torch.from_numpy(elem).unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(elem, torch.Tensor):
            return elem.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(elem, str):
            return [[elem]]
        else:
            return elem

    # For online
    def merge_input(self, inputs) -> torch.tensor:
        """Abstract method to merge multiple inputs.

        Subclasses should implement this if they support merging inputs for, e.g.,
        batched inference across multiple environments.

        :param inputs: The inputs to merge.
        :returns: The merged input tensor.
        :rtype: torch.tensor
        :raises NotImplementedError: if not implemented by the subclass.
        """
        raise NotImplementedError

    def merge_state(self, states) -> Optional[List[torch.Tensor]]:
        """Abstract method to merge multiple states.

        Subclasses should implement this if they support merging states.

        :param states: The states to merge.
        :returns: The merged state as an optional list of tensors.
        :rtype: Optional[List[torch.Tensor]]
        :raises NotImplementedError: if not implemented by the subclass.
        """
        raise NotImplementedError

    def split_state(self, state, split_num) -> Optional[List[List[torch.Tensor]]]:
        """Abstract method to split a state into multiple states.

        Subclasses should implement this if they support splitting states.

        :param state: The state to split.
        :param split_num: The number of ways to split the state.
        :returns: An optional list of split states.
        :rtype: Optional[List[List[torch.Tensor]]]
        :raises NotImplementedError: if not implemented by the subclass.
        """
        raise NotImplementedError

    def split_action(self, action, split_num) -> Optional[List[Dict[str, torch.Tensor]]]:
        """Splits a batched action into a list of individual actions.

        Handles actions as dictionaries of tensors, single tensors, or lists.
        Converts tensors to numpy arrays on CPU.

        :param action: The batched action to split.
        :param split_num: The number of individual actions in the batch.
        :returns: A list of individual actions, or the original action list if already a list.
        :rtype: Optional[List[Dict[str, torch.Tensor]]]
        :raises NotImplementedError: if the action type is not supported.
        """
        if isinstance(action, dict):
            # for k, v in action.items():
            #     action[k] = v.view(-1,1)
            result_actions = [{k: v[i].cpu().numpy() for k, v in action.items()} for i in range(0, split_num)]
            return result_actions
        elif isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
            result_actions = [action_np[i] for i in range(0, split_num)]
            return result_actions
        elif isinstance(action, list):
            return action
        else:
            raise NotImplementedError

