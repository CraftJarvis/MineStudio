from typing import Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from minestudio.utils.vpt_lib.action_head import fan_in_linear
from minestudio.utils.vpt_lib.normalize_ewma import NormalizeEwma


class ScaledMSEHead(nn.Module):
    """
    A linear output layer that normalizes its targets to have a mean of 0 and a standard deviation of 1.
    This is achieved by using an Exponential Moving Average (EWMA) normalizer on the targets
    before calculating the Mean Squared Error (MSE) loss. The predictions are made in the
    original (unnormalized) space, but the loss is computed in the normalized space.
    """

    def __init__(
        self, input_size: int, output_size: int, norm_type: Optional[str] = "ewma", norm_kwargs: Optional[Dict] = None
    ):
        """
        Initializes the ScaledMSEHead.

        :param input_size: The dimensionality of the input features.
        :type input_size: int
        :param output_size: The dimensionality of the output (action) space.
        :type output_size: int
        :param norm_type: The type of normalizer to use. Currently, only "ewma" is implicitly supported
                          as `NormalizeEwma` is directly instantiated.
        :type norm_type: Optional[str]
        :param norm_kwargs: Keyword arguments to pass to the normalizer (`NormalizeEwma`).
        :type norm_kwargs: Optional[Dict]
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.norm_type = norm_type

        self.linear = nn.Linear(self.input_size, self.output_size)

        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.normalizer = NormalizeEwma(output_size, **norm_kwargs)

    def reset_parameters(self):
        """
        Initializes the weights of the linear layer and resets the normalizer's parameters.
        The linear layer's weights are initialized orthogonally and then scaled using fan-in initialization.
        """
        init.orthogonal_(self.linear.weight)
        fan_in_linear(self.linear)
        self.normalizer.reset_parameters()

    def forward(self, input_data):
        """
        Performs a forward pass through the linear layer.

        :param input_data: The input tensor.
        :type input_data: torch.Tensor
        :returns: The output of the linear layer (predictions in the original space).
        :rtype: torch.Tensor
        """
        return self.linear(input_data)

    def loss(self, prediction, target, reduction="mean"):
        """
        Calculates the Mean Squared Error (MSE) loss between the prediction and the target.
        The target is first normalized using the internal EWMA normalizer.
        The loss is computed in this normalized space.

        :param prediction: The predicted output from the forward pass (in original space).
        :type prediction: torch.Tensor
        :param target: The target values (in original space).
        :type target: torch.Tensor
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                          'none': no reduction will be applied,
                          'mean': the sum of the output will be divided by the number of elements in the output,
                          'sum': the output will be summed.
        :type reduction: str
        :returns: The MSE loss.
        :rtype: torch.Tensor
        """
        return F.mse_loss(prediction, self.normalizer(target), reduction=reduction)

    def denormalize(self, input_data):
        """
        Converts an input value from the normalized space back into the original space
        using the inverse operation of the internal normalizer.

        :param input_data: The data in the normalized space.
        :type input_data: torch.Tensor
        :returns: The data in the original space.
        :rtype: torch.Tensor
        """
        return self.normalizer.denormalize(input_data)

    def normalize(self, input_data):
        """
        Normalizes the input data using the internal EWMA normalizer.

        :param input_data: The data in the original space.
        :type input_data: torch.Tensor
        :returns: The data in the normalized space.
        :rtype: torch.Tensor
        """
        return self.normalizer(input_data)
