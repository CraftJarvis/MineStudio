import random
import logging
from typing import Any, Tuple, Dict, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
import gym
import gymnasium
from gym3.types import DictType, Discrete, Real, TensorType, ValType

LOG0 = -100

def fan_in_linear(module: nn.Module, scale=1.0, bias=True):
    """
    Initializes the weights of a linear module using fan-in initialization.
    The weights are scaled by `scale / norm`, where norm is the L2 norm of the weights.
    Biases are initialized to zero if `bias` is True.

    :param module: The linear module to initialize.
    :type module: nn.Module
    :param scale: The scaling factor for the weights.
    :type scale: float
    :param bias: Whether to initialize biases to zero.
    :type bias: bool
    """
    module.weight.data *= scale / module.weight.norm(dim=1, p=2, keepdim=True)

    if bias:
        module.bias.data *= 0

class ActionHead(nn.Module):
    """
    Abstract base class for action heads.
    Action heads are responsible for converting network outputs into action probability distributions
    and providing methods for sampling, calculating log probabilities, entropy, and KL divergence.
    """

    def forward(self, input_data, **kwargs) -> Any:
        """
        Performs a forward pass through the action head.

        :param input_data: The input tensor from the policy network.
        :type input_data: torch.Tensor
        :param \\*\\*kwargs: Additional keyword arguments.
        :returns: Parameters describing the probability distribution of actions.
        :rtype: Any
        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def logprob(self, action_sample, pd_params, **kwargs):
        """
        Calculates the logarithm of the probability of sampling `action_sample`
        from a probability distribution described by `pd_params`.

        :param action_sample: The sampled action.
        :type action_sample: Any
        :param pd_params: Parameters describing the probability distribution.
        :type pd_params: Any
        :param \\*\\*kwargs: Additional keyword arguments.
        :returns: The log probability of the action sample.
        :rtype: torch.Tensor
        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def entropy(self, pd_params):
        """
        Calculates the entropy of the probability distribution described by `pd_params`.

        :param pd_params: Parameters describing the probability distribution.
        :type pd_params: Any
        :returns: The entropy of the distribution.
        :rtype: torch.Tensor
        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def sample(self, pd_params, deterministic: bool = False) -> Any:
        """
        Draws a sample from the probability distribution given by `pd_params`.

        :param pd_params: Parameters of a probability distribution.
        :type pd_params: Any
        :param deterministic: Whether to return a stochastic sample or the deterministic mode of the distribution.
        :type deterministic: bool
        :returns: A sampled action.
        :rtype: Any
        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def kl_divergence(self, params_q, params_p):
        """
        Calculates the KL divergence between two distributions described by `params_q` and `params_p`.
        KL(Q || P).

        :param params_q: Parameters of the first distribution (Q).
        :type params_q: Any
        :param params_p: Parameters of the second distribution (P).
        :type params_p: Any
        :returns: The KL divergence between the two distributions.
        :rtype: torch.Tensor
        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class DiagGaussianActionHead(ActionHead):
    """
    Action head for normally distributed, uncorrelated continuous actions.
    Means are predicted by a linear layer, while standard deviations are learnable parameters.
    """

    LOG2PI = np.log(2.0 * np.pi)

    def __init__(self, input_dim: int, num_dimensions: int):
        """
        Initializes the DiagGaussianActionHead.

        :param input_dim: The dimensionality of the input features.
        :type input_dim: int
        :param num_dimensions: The number of dimensions of the action space.
        :type num_dimensions: int
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_dimensions = num_dimensions

        self.linear_layer = nn.Linear(input_dim, num_dimensions)
        self.log_std = nn.Parameter(torch.zeros(num_dimensions), requires_grad=True)

    def reset_parameters(self):
        """Initializes the weights of the linear layer and biases."""
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data: torch.Tensor, mask=None, **kwargs) -> torch.Tensor:
        """
        Computes the means and log standard deviations of the Gaussian distribution.

        :param input_data: The input tensor from the policy network.
        :type input_data: torch.Tensor
        :param mask: An optional mask (not used in this head).
        :type mask: Optional[torch.Tensor]
        :param \\*\\*kwargs: Additional keyword arguments.
        :returns: A tensor where the last dimension contains means and log_stds.
                  Shape: (..., num_dimensions, 2)
        :rtype: torch.Tensor
        :raises AssertionError: If a mask is provided.
        """
        assert not mask, "Can not use a mask in a gaussian action head"
        means = self.linear_layer(input_data)
        # Unsqueeze many times to get to the same shape
        logstd = self.log_std[(None,) * (len(means.shape) - 1)]

        mean_view, logstd = torch.broadcast_tensors(means, logstd)

        return torch.stack([mean_view, logstd], dim=-1)

    def logprob(self, action_sample: torch.Tensor, pd_params: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-likelihood of `action_sample` given the distribution parameters.
        The distribution is a multivariate Gaussian with a diagonal covariance matrix.

        :param action_sample: The sampled actions. Shape: (..., num_dimensions)
        :type action_sample: torch.Tensor
        :param pd_params: Parameters of the Gaussian distribution (means and log_stds).
                          Shape: (..., num_dimensions, 2)
        :type pd_params: torch.Tensor
        :returns: The log probability of the action samples. Shape: (...)
        :rtype: torch.Tensor
        """
        means = pd_params[..., 0]
        log_std = pd_params[..., 1]

        std = torch.exp(log_std)

        z_score = (action_sample - means) / std

        return -(0.5 * ((z_score ** 2 + self.LOG2PI).sum(dim=-1)) + log_std.sum(dim=-1))

    def entropy(self, pd_params: torch.Tensor) -> torch.Tensor:
        """
        Calculates the entropy of the Gaussian distribution.
        For a diagonal Gaussian, entropy is 0.5 * sum(log(2 * pi * e * sigma_i^2)).

        :param pd_params: Parameters of the Gaussian distribution (means and log_stds).
                          Shape: (..., num_dimensions, 2)
        :type pd_params: torch.Tensor
        :returns: The entropy of the distribution. Shape: (...)
        :rtype: torch.Tensor
        """
        log_std = pd_params[..., 1]
        return (log_std + 0.5 * (self.LOG2PI + 1)).sum(dim=-1)

    def sample(self, pd_params: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Samples an action from the Gaussian distribution.

        :param pd_params: Parameters of the Gaussian distribution (means and log_stds).
                          Shape: (..., num_dimensions, 2)
        :type pd_params: torch.Tensor
        :param deterministic: If True, returns the mean (mode) of the distribution.
                              If False, returns a stochastic sample.
        :type deterministic: bool
        :returns: A sampled action. Shape: (..., num_dimensions)
        :rtype: torch.Tensor
        """
        means = pd_params[..., 0]
        log_std = pd_params[..., 1]

        if deterministic:
            return means
        else:
            return torch.randn_like(means) * torch.exp(log_std) + means

    def kl_divergence(self, params_q: torch.Tensor, params_p: torch.Tensor) -> torch.Tensor:
        """
        Calculates the KL divergence KL(Q || P) between two diagonal Gaussian distributions Q and P.
        Formula: log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2) / (2 * sigma_p^2) - 0.5, summed over dimensions.

        :param params_q: Parameters of the first Gaussian distribution Q (means_q, log_std_q).
                         Shape: (..., num_dimensions, 2)
        :type params_q: torch.Tensor
        :param params_p: Parameters of the second Gaussian distribution P (means_p, log_std_p).
                         Shape: (..., num_dimensions, 2)
        :type params_p: torch.Tensor
        :returns: The KL divergence. Shape: (..., 1)
        :rtype: torch.Tensor
        """
        means_q = params_q[..., 0]
        log_std_q = params_q[..., 1]

        means_p = params_p[..., 0]
        log_std_p = params_p[..., 1]

        std_q = torch.exp(log_std_q)
        std_p = torch.exp(log_std_p)

        kl_div = log_std_p - log_std_q + (std_q ** 2 + (means_q - means_p) ** 2) / (2.0 * std_p ** 2) - 0.5

        return kl_div.sum(dim=-1, keepdim=True)


class CategoricalActionHead(ActionHead):
    """
    Action head for categorical (discrete) actions.
    It uses a linear layer to produce logits for each action.
    Supports temperature scaling and nucleus sampling.
    """

    def __init__(
        self,
        input_dim: int,
        shape: Tuple[int],
        num_actions: int,
        builtin_linear_layer: bool = True,
        temperature: float = 1.0,
        nucleus_prob: Optional[float] = None,
    ):
        """
        Initializes the CategoricalActionHead.

        :param input_dim: The dimensionality of the input features.
        :type input_dim: int
        :param shape: The shape of the action space, excluding the number of actions dimension.
                      For example, if action space is (H, W, num_actions), shape is (H, W).
        :type shape: Tuple[int]
        :param num_actions: The number of possible discrete actions.
        :type num_actions: int
        :param builtin_linear_layer: Whether to include a linear layer to map input_dim to num_actions.
                                     If False, input_dim must equal num_actions.
        :type builtin_linear_layer: bool
        :param temperature: Temperature for scaling logits before softmax. Higher temperature -> softer distribution.
        :type temperature: float
        :param nucleus_prob: Probability threshold for nucleus sampling. If None, vanilla sampling is used.
        :type nucleus_prob: Optional[float]
        :raises AssertionError: If `builtin_linear_layer` is False and `input_dim` != `num_actions`.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions
        self.output_shape = shape + (num_actions,)
        self.temperature = temperature
        self.nucleus_prob = nucleus_prob

        if builtin_linear_layer:
            self.linear_layer = nn.Linear(input_dim, np.prod(self.output_shape))
        else:
            assert (
                input_dim == num_actions
            ), f"If input_dim ({input_dim}) != num_actions ({num_actions}), you need a linear layer to convert them."
            self.linear_layer = None

    def reset_parameters(self):
        """Initializes the weights of the linear layer (if it exists) and biases."""
        if self.linear_layer is not None:
            init.orthogonal_(self.linear_layer.weight, gain=0.01)
            init.constant_(self.linear_layer.bias, 0.0)
            fan_in_linear(self.linear_layer, scale=0.01) # Corrected: removed finit prefix

    def forward(self, input_data: torch.Tensor, mask=None, **kwargs) -> Any:
        """
        Computes the log probabilities (logits) for each action.
        Applies temperature scaling and masking if provided.

        :param input_data: The input tensor from the policy network.
        :type input_data: torch.Tensor
        :param mask: An optional boolean mask. Logits for masked-out actions are set to a very small number (LOG0).
                     Shape should be broadcastable to the logits shape before the num_actions dimension.
        :type mask: Optional[torch.Tensor]
        :param \\*\\*kwargs: Additional keyword arguments.
        :returns: Logits for each action after processing. Shape: (..., *self.output_shape)
        :rtype: torch.Tensor
        """
        if self.linear_layer is not None:
            flat_out = self.linear_layer(input_data)
        else:
            flat_out = input_data
        shaped_out = flat_out.reshape(flat_out.shape[:-1] + self.output_shape)
        shaped_out /= self.temperature
        if mask is not None:
            shaped_out[~mask] = LOG0

        # Convert to float32 to avoid RuntimeError: "log_softmax_lastdim_kernel_impl" not implemented for 'Half'
        return F.log_softmax(shaped_out.float(), dim=-1)

    def logprob(self, actions: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log probability of the given `actions` based on the `logits`.
        It gathers the log probabilities corresponding to the chosen actions and sums them
        if the action space has multiple dimensions (e.g., for MultiDiscrete).

        :param actions: The sampled actions. Expected to be long type.
                        Shape: (..., *self.output_shape[:-1])
        :type actions: torch.Tensor
        :param logits: The log probabilities (output of the forward pass).
                       Shape: (..., *self.output_shape)
        :type logits: torch.Tensor
        :returns: The sum of log probabilities for the chosen actions. Shape: (...)
        :rtype: torch.Tensor
        """
        value = actions.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]
        result = log_pmf.gather(-1, value).squeeze(-1)
        # result is per-entry, still of size self.output_shape[:-1]; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            result = result.sum(dim=-1)
        return result

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates the entropy of the categorical distribution defined by `logits`.
        Entropy = - sum(probs * log_probs).
        The result is summed if the action space has multiple dimensions.

        :param logits: The log probabilities (output of the forward pass).
                       Shape: (..., *self.output_shape)
        :type logits: torch.Tensor
        :returns: The entropy of the distribution. Shape: (...)
        :rtype: torch.Tensor
        """
        probs = torch.exp(logits)
        entropy = -torch.sum(probs * logits, dim=-1)
        # entropy is per-entry, still of size self.output_shape[:-1]; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            entropy = entropy.sum(dim=-1)
        return entropy

    # minecraft domain should directly use this sample function
    def vanilla_sample(self, logits: torch.Tensor, deterministic: bool = False, **kwargs) -> Any:
        """
        Samples an action from the categorical distribution using the Gumbel-Max trick for stochastic sampling,
        or argmax for deterministic sampling. This is the original sampling method from the VPT library.

        :param logits: The log probabilities for each action.
                       Shape: (..., *self.output_shape)
        :type logits: torch.Tensor
        :param deterministic: If True, returns the action with the highest logit (argmax).
                              If False, returns a stochastic sample using Gumbel-Max.
        :type deterministic: bool
        :param \\*\\*kwargs: Additional keyword arguments (not used).
        :returns: A sampled action. Shape: (..., *self.output_shape[:-1])
        :rtype: torch.Tensor
        """
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            # Gumbel-Softmax trick.
            u = torch.rand_like(logits)
            # In float16, if you have around 2^{float_mantissa_bits} logits, sometimes you'll sample 1.0
            # Then the log(-log(1.0)) will give -inf when it should give +inf
            # This is a silly hack to get around that.
            # This hack does not skew the probability distribution, because this event can't possibly win the argmax.
            u[u == 1.0] = 0.999
            
            return torch.argmax(logits - torch.log(-torch.log(u)), dim=-1)
    
    def nucleus_sample(self, logits: torch.Tensor, deterministic: bool = False, p: float = 0.85, **kwargs) -> Any:
        """
        Samples an action using nucleus (top-p) sampling.
        It considers the smallest set of actions whose cumulative probability exceeds `p`.
        If deterministic, falls back to vanilla sampling with determinism.

        :param logits: The log probabilities for each action.
                       Shape: (..., *self.output_shape)
        :type logits: torch.Tensor
        :param deterministic: If True, uses vanilla deterministic sampling.
        :type deterministic: bool
        :param p: The cumulative probability threshold for nucleus sampling.
        :type p: float
        :param \\*\\*kwargs: Additional keyword arguments (passed to vanilla_sample if deterministic).
        :returns: A sampled action. Shape: (..., *self.output_shape[:-1])
        :rtype: torch.Tensor
        """
        if deterministic:
            return self.vanilla_sample(logits, deterministic)
        probs = torch.exp(logits)
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1) 
        nucleus = cum_sum_probs < p 
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
        sorted_log_probs = torch.log(sorted_probs)
        sorted_log_probs[~nucleus] = float('-inf')
        sampled_sorted_indexes = self.vanilla_sample(sorted_log_probs, deterministic=False)
        res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))
        return res.squeeze(-1)

    def sample(self, logits: torch.Tensor, deterministic: bool = False, **kwargs) -> Any:
        """
        Samples an action from the categorical distribution.
        Uses nucleus sampling if `self.nucleus_prob` is set, otherwise uses vanilla sampling.

        :param logits: The log probabilities for each action.
                       Shape: (..., *self.output_shape)
        :type logits: torch.Tensor
        :param deterministic: If True, returns the most likely action.
                              If False, returns a stochastic sample.
        :type deterministic: bool
        :param \\*\\*kwargs: Additional keyword arguments for the specific sampling method.
        :returns: A sampled action. Shape: (..., *self.output_shape[:-1])
        :rtype: torch.Tensor
        """
        if self.nucleus_prob is None:
            return self.vanilla_sample(logits, deterministic, **kwargs)
        else:
            return self.nucleus_sample(logits, deterministic, p=self.nucleus_prob, **kwargs)

    def kl_divergence(self, logits_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
        """
        Calculates the KL divergence KL(Q || P) between two categorical distributions Q and P,
        defined by their logits.
        Formula: sum(exp(Q_i) * (Q_i - P_i)).
        The result is summed if the action space has multiple dimensions.

        :param logits_q: Logits of the first distribution Q.
                         Shape: (..., *self.output_shape)
        :type logits_q: torch.Tensor
        :param logits_p: Logits of the second distribution P.
                         Shape: (..., *self.output_shape)
        :type logits_p: torch.Tensor
        :returns: The KL divergence. Shape: (..., 1)
        :rtype: torch.Tensor
        """
        kl = (torch.exp(logits_q) * (logits_q - logits_p)).sum(-1, keepdim=True)
        # kl is per-entry, still of size self.output_shape; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            kl = kl.sum(dim=-2)  # dim=-2 because we use keepdim=True above.
        return kl

class MSEActionHead(ActionHead):
    """
    Action head for continuous actions where the loss is Mean Squared Error (MSE)
    between the predicted actions (means) and the target actions.
    This head essentially predicts the mean of a distribution with fixed, infinitesimal variance.
    """

    def __init__(self, input_dim: int, num_dimensions: int):
        """
        Initializes the MSEActionHead.

        :param input_dim: The dimensionality of the input features.
        :type input_dim: int
        :param num_dimensions: The number of dimensions of the action space.
        :type num_dimensions: int
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_dimensions = num_dimensions

        self.linear_layer = nn.Linear(input_dim, num_dimensions)

    def reset_parameters(self):
        """Initializes the weights of the linear layer and biases."""
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data: torch.Tensor, mask=None, **kwargs) -> torch.Tensor:
        """
        Computes the predicted mean actions using a linear layer.

        :param input_data: The input tensor from the policy network.
        :type input_data: torch.Tensor
        :param mask: An optional mask (not used in this head).
        :type mask: Optional[torch.Tensor]
        :param \\*\\*kwargs: Additional keyword arguments.
        :returns: The predicted mean actions. Shape: (..., num_dimensions)
        :rtype: torch.Tensor
        :raises AssertionError: If a mask is provided.
        """
        assert not mask, "Can not use a mask in a mse action head"
        means = self.linear_layer(input_data)

        return means

    def logprob(self, action_sample: torch.Tensor, pd_params: torch.Tensor) -> torch.Tensor:
        """
        Calculates a pseudo log-probability, which is the negative squared error.
        This is not a true log-probability but is used for compatibility in some RL frameworks.

        :param action_sample: The target actions. Shape: (..., num_dimensions)
        :type action_sample: torch.Tensor
        :param pd_params: The predicted mean actions (output of the forward pass).
                          Shape: (..., num_dimensions)
        :type pd_params: torch.Tensor
        :returns: The negative sum of squared errors. Shape: (...)
        :rtype: torch.Tensor
        """
        return - ((action_sample - pd_params).pow(2)).sum(dim=-1)

    def entropy(self, pd_params: torch.Tensor) -> torch.Tensor:
        """
        Returns zero entropy, as this head represents a deterministic prediction (delta distribution).

        :param pd_params: The predicted mean actions.
        :type pd_params: torch.Tensor
        :returns: A tensor of zeros with the same batch shape as pd_params. Shape: (...)
        :rtype: torch.Tensor
        """
        return torch.zeros_like(pd_params).sum(dim=-1)

    def sample(self, pd_params: torch.Tensor, deterministic: bool = False, **kwargs) -> torch.Tensor:
        """
        Returns the predicted mean actions, as this head is deterministic.

        :param pd_params: The predicted mean actions (output of the forward pass).
                          Shape: (..., num_dimensions)
        :type pd_params: torch.Tensor
        :param deterministic: Ignored, as sampling is always deterministic.
        :type deterministic: bool
        :param \\*\\*kwargs: Additional keyword arguments (not used).
        :returns: The predicted mean actions. Shape: (..., num_dimensions)
        :rtype: torch.Tensor
        """
        return pd_params

    def kl_divergence(self, params_q: torch.Tensor, params_p: torch.Tensor) -> torch.Tensor:
        """
        KL divergence is not well-defined for this action head in a general sense
        as it represents a delta distribution.

        :param params_q: Parameters of the first distribution.
        :type params_q: torch.Tensor
        :param params_p: Parameters of the second distribution.
        :type params_p: torch.Tensor
        :raises NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("KL divergence is not defined for MSE action head")

class TupleActionHead(nn.ModuleList, ActionHead):
    """
    An action head that combines multiple sub-action heads, where actions are structured as a tuple.
    Each element of the tuple corresponds to an action from a sub-head.
    Inherits from `nn.ModuleList` to manage sub-heads and `ActionHead` for the interface.
    """

    def reset_parameters(self):
        """Calls `reset_parameters` on each sub-head."""
        for subhead in self:
            subhead.reset_parameters()

    def forward(self, input_data: torch.Tensor, **kwargs) -> Any:
        """
        Passes the input data through each sub-head and returns a tuple of their outputs.

        :param input_data: The input tensor from the policy network.
        :type input_data: torch.Tensor
        :param \\*\\*kwargs: Additional keyword arguments (passed to each sub-head).
        :returns: A tuple where each element is the output (pd_params) of a sub-head.
        :rtype: Tuple[Any, ...]
        """
        return tuple([ subhead(input_data, **kwargs) for subhead in self ]) # Added **kwargs

    def logprob(self, actions: Tuple[torch.Tensor], logits: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the log probability for each action in the tuple using the corresponding sub-head
        and its logits. Returns a tuple of log probabilities.

        :param actions: A tuple of sampled actions, one for each sub-head.
        :type actions: Tuple[torch.Tensor, ...]
        :param logits: A tuple of probability distribution parameters (e.g., logits) from each sub-head.
        :type logits: Tuple[torch.Tensor, ...]
        :returns: A tuple of log probabilities, one for each sub-action.
        :rtype: Tuple[torch.Tensor, ...]
        """
        return tuple([ subhead.logprob(actions[k], logits[k]) for k, subhead in enumerate(self) ])

    def sample(self, logits: Tuple[torch.Tensor], deterministic: bool = False) -> Any:
        """
        Samples an action from each sub-head and returns a tuple of these actions.

        :param logits: A tuple of probability distribution parameters from each sub-head.
        :type logits: Tuple[torch.Tensor, ...]
        :param deterministic: Whether to perform deterministic sampling for each sub-head.
        :type deterministic: bool
        :returns: A tuple of sampled actions.
        :rtype: Tuple[Any, ...]
        """
        return tuple([ subhead.sample(logits[k], deterministic) for k, subhead in enumerate(self) ])

    def entropy(self, logits: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the entropy for each sub-distribution and returns a tuple of entropies.

        :param logits: A tuple of probability distribution parameters from each sub-head.
        :type logits: Tuple[torch.Tensor, ...]
        :returns: A tuple of entropies, one for each sub-distribution.
        :rtype: Tuple[torch.Tensor, ...]
        """
        return tuple([ subhead.entropy(logits[k]) for k, subhead in enumerate(self) ])

    def kl_divergence(self, logits_q: Tuple[torch.Tensor], logits_p: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the KL divergence for each pair of sub-distributions (Q_k || P_k)
        and returns their sum.

        :param logits_q: A tuple of parameters for the first set of distributions (Q).
        :type logits_q: Tuple[torch.Tensor, ...]
        :param logits_p: A tuple of parameters for the second set of distributions (P).
        :type logits_p: Tuple[torch.Tensor, ...]
        :returns: The sum of KL divergences from all sub-heads.
        :rtype: torch.Tensor
        """
        return sum( subhead.kl_divergence(logits_q[k], logits_p[k]) for k, subhead in enumerate(self) )

class DictActionHead(nn.ModuleDict, ActionHead):
    """
    An action head that combines multiple sub-action heads, where actions are structured as a dictionary.
    Each key-value pair in the dictionary corresponds to an action from a named sub-head.
    Inherits from `nn.ModuleDict` to manage sub-heads and `ActionHead` for the interface.
    """

    def reset_parameters(self):
        """Calls `reset_parameters` on each sub-head in the dictionary."""
        for subhead in self.values():
            subhead.reset_parameters()

    def forward(self, input_data: torch.Tensor, **kwargs) -> Any:
        """
        Passes input data through each sub-head. Allows passing specific keyword arguments
        to individual sub-heads based on their keys.

        Example:
        If this ModuleDict has submodules keyed by 'A', 'B', and 'C', we could call:
        `forward(input_data, foo={'A': True, 'C': False}, bar={'A': 7})`
        Then children will be called with:
            A: `subhead_A(input_data, foo=True, bar=7)`
            B: `subhead_B(input_data)`
            C: `subhead_C(input_data, foo=False)`

        :param input_data: The input tensor from the policy network.
        :type input_data: torch.Tensor
        :param \\*\\*kwargs: Keyword arguments. If a kwarg's value is a dictionary, its items
                             are passed to sub-heads matching the keys.
        :returns: A dictionary where keys are sub-head names and values are their outputs (pd_params).
        :rtype: Dict[str, Any]
        """
        result = {}
        for head_name, subhead in self.items():
            head_kwargs = {
                kwarg_name: kwarg[head_name]
                for kwarg_name, kwarg in kwargs.items()
                if kwarg is not None and head_name in kwarg
            }
            result[head_name] = subhead(input_data, **head_kwargs)
        return result

    def logprob(self, actions: Dict[str, torch.Tensor], logits: Dict[str, torch.Tensor], return_dict=False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates log probabilities for actions from each sub-head.
        Can return a dictionary of log probabilities or their sum.

        :param actions: A dictionary of sampled actions, keyed by sub-head name.
        :type actions: Dict[str, torch.Tensor]
        :param logits: A dictionary of pd_params from each sub-head, keyed by sub-head name.
        :type logits: Dict[str, torch.Tensor]
        :param return_dict: If True, returns a dictionary of log probabilities.
                            If False, returns the sum of log probabilities.
        :type return_dict: bool
        :returns: Either a sum of log probabilities (Tensor) or a dictionary of log probabilities.
        :rtype: Union[torch.Tensor, Dict[str, torch.Tensor]]
        """
        if return_dict:
            return {k: subhead.logprob(actions[k], logits[k]) for k, subhead in self.items()}
        else:
            return sum(subhead.logprob(actions[k], logits[k]) for k, subhead in self.items())

    def sample(self, logits: Dict[str, torch.Tensor], deterministic: bool = False) -> Any:
        """
        Samples an action from each sub-head and returns a dictionary of these actions.

        :param logits: A dictionary of pd_params from each sub-head, keyed by sub-head name.
        :type logits: Dict[str, torch.Tensor]
        :param deterministic: Whether to perform deterministic sampling for each sub-head.
        :type deterministic: bool
        :returns: A dictionary of sampled actions, keyed by sub-head name.
        :rtype: Dict[str, Any]
        """
        return {k: subhead.sample(logits[k], deterministic) for k, subhead in self.items()}

    def entropy(self, logits: Dict[str, torch.Tensor], return_dict=False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates the entropy for each sub-distribution.
        Can return a dictionary of entropies or their sum.

        :param logits: A dictionary of pd_params from each sub-head, keyed by sub-head name.
        :type logits: Dict[str, torch.Tensor]
        :param return_dict: If True, returns a dictionary of entropies.
                            If False, returns the sum of entropies.
        :type return_dict: bool
        :returns: Either a sum of entropies (Tensor) or a dictionary of entropies.
        :rtype: Union[torch.Tensor, Dict[str, torch.Tensor]]
        """
        if return_dict:
            return {k: subhead.entropy(logits[k]) for k, subhead in self.items()}
        else:
            return sum(subhead.entropy(logits[k]) for k, subhead in self.items())

    def kl_divergence(self, logits_q: Dict[str, torch.Tensor], logits_p: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the KL divergence for each pair of sub-distributions (Q_k || P_k)
        and returns their sum.

        :param logits_q: A dictionary of parameters for the first set of distributions (Q).
        :type logits_q: Dict[str, torch.Tensor]
        :param logits_p: A dictionary of parameters for the second set of distributions (P).
        :type logits_p: Dict[str, torch.Tensor]
        :returns: The sum of KL divergences from all sub-heads.
        :rtype: torch.Tensor
        """
        return sum(subhead.kl_divergence(logits_q[k], logits_p[k]) for k, subhead in self.items())

def make_action_head(ac_space: ValType, pi_out_size: int, temperature: float = 1.0, **kwargs):
    """
    Helper function to create an appropriate action head based on the action space type.
    Supports `gymnasium.spaces` and some `gym.spaces`.

    :param ac_space: The action space of the environment.
    :type ac_space: Union[gymnasium.spaces.Space, gym.spaces.Space, ValType]
    :param pi_out_size: The output size of the policy network feature extractor.
    :type pi_out_size: int
    :param temperature: Temperature for categorical action heads.
    :type temperature: float
    :param \\*\\*kwargs: Additional keyword arguments to pass to the action head constructor.
    :returns: An initialized action head.
    :rtype: ActionHead
    :raises NotImplementedError: If the action space type is not supported.
    """
    if isinstance(ac_space, gymnasium.spaces.MultiDiscrete):
        return CategoricalActionHead(pi_out_size, ac_space.shape, ac_space.nvec[0].item(), temperature=temperature, **kwargs)
    elif isinstance(ac_space, gymnasium.spaces.Dict):
        return DictActionHead({k: make_action_head(v, pi_out_size, temperature, **kwargs) for k, v in ac_space.items()})
    elif isinstance(ac_space, gymnasium.spaces.Tuple):
        return TupleActionHead([make_action_head(v, pi_out_size, temperature, **kwargs) for v in ac_space])
    elif isinstance(ac_space, gym.spaces.Discrete):
        return CategoricalActionHead(pi_out_size, ac_space.shape, ac_space.n, temperature=temperature, **kwargs)
    elif isinstance(ac_space, gym.spaces.Box) or isinstance(ac_space, gymnasium.spaces.Box):
        assert len(ac_space.shape) == 1, "Nontrivial shapes not yet implemented."
        return MSEActionHead(pi_out_size, ac_space.shape[0], **kwargs)
    raise NotImplementedError(f"Action space of type {type(ac_space)} is not supported")

# def make_action_head(ac_space: ValType, pi_out_size: int, temperature: float = 1.0):
# """
# Helper function to create an action head corresponding to the environment action space (gym3.types version).

# :param ac_space: The action space of the environment, typically from gym3.
# :type ac_space: ValType
# :param pi_out_size: The output size of the policy network feature extractor.
# :type pi_out_size: int
# :param temperature: Temperature for categorical action heads.
# :type temperature: float
# :returns: An initialized action head.
# :rtype: ActionHead
# :raises NotImplementedError: If the action space type is not supported.
# :raises AssertionError: If conditions for specific heads are not met (e.g., shape for DiagGaussian).
# """


