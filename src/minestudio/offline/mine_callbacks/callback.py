'''
Date: 2024-11-12 10:57:29
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2025-01-18 13:53:17
FilePath: /MineStudio/minestudio/offline/mine_callbacks/callback.py
'''
import torch
from typing import Dict, Any
from minestudio.models import MinePolicy

class ObjectiveCallback:
    """
    Base class for objective callbacks used in MineLightning training.

    Objective callbacks are used to define and calculate specific loss components
    or metrics during the training or validation step. Subclasses should implement
    the `__call__` method to compute their specific objective.
    """

    def __init__(self):
        """
        Initializes the ObjectiveCallback.
        """
        ...

    def __call__(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        step_name: str, 
        latents: Dict[str, torch.Tensor], 
        mine_policy: MinePolicy
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the objective value(s).

        This method should be implemented by subclasses to compute their specific
        loss or metric. The returned dictionary will be aggregated with results
        from other callbacks.

        :param batch: A dictionary containing the batch data.
        :type batch: Dict[str, Any]
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :param step_name: The name of the current step (e.g., 'train', 'val').
        :type step_name: str
        :param latents: A dictionary containing the policy's latent outputs.
        :type latents: Dict[str, torch.Tensor]
        :param mine_policy: The MinePolicy model.
        :type mine_policy: MinePolicy
        :returns: A dictionary where keys are metric names and values are the
                  corresponding tensor values. An empty dictionary if no objective
                  is computed.
        :rtype: Dict[str, torch.Tensor]
        """
        return {}

    def before_step(self, batch, batch_idx, step_name):
        """
        A hook called before the main batch step processing.

        This can be used to modify the batch or perform other actions before
        the model forward pass and objective calculations.

        :param batch: The input batch data.
        :type batch: Dict[str, Any]
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :param step_name: The name of the current step (e.g., 'train', 'val').
        :type step_name: str
        :returns: The (potentially modified) batch data.
        :rtype: Dict[str, Any]
        """
        return batch
