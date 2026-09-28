'''
Date: 2024-11-12 13:59:08
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-09 15:51:34
FilePath: /MineStudio/minestudio/train/mine_callbacks/behavior_clone.py
'''

import torch
from typing import Dict, Any
from minestudio.models import MinePolicy
from minestudio.offline.mine_callbacks.callback import ObjectiveCallback

class BehaviorCloneCallback(ObjectiveCallback):
    """
    A callback for behavior cloning.

    This callback calculates the behavior cloning loss, which is the negative
    log-likelihood of the agent's actions under the policy's action distribution.
    It also calculates the entropy of the policy's action distribution.
    """
        
    def __init__(self, weight: float=1.0):
        """
        Initializes the BehaviorCloneCallback.

        :param weight: The weight to apply to the behavior cloning loss. Defaults to 1.0.
        :type weight: float
        """
        super().__init__()
        self.weight = weight

    def __call__(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        step_name: str, 
        latents: Dict[str, torch.Tensor], 
        mine_policy: MinePolicy, 
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the behavior cloning loss and entropy.

        The loss is computed as the negative log-probability of the agent's actions
        (both camera and buttons) under the current policy. A mask is applied to
        ignore padding in camera actions. The entropy of the policy's action
        distribution is also computed.

        :param batch: A dictionary containing the batch data. Must include 'agent_action'.
        :type batch: Dict[str, Any]
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :param step_name: The name of the current step (e.g., 'train', 'val').
        :type step_name: str
        :param latents: A dictionary containing the policy's latent outputs, including 'pi_logits'.
        :type latents: Dict[str, torch.Tensor]
        :param mine_policy: The MinePolicy model.
        :type mine_policy: MinePolicy
        :returns: A dictionary containing the calculated losses and metrics:
                  'loss': The weighted behavior cloning loss.
                  'camera_loss': The camera action loss.
                  'button_loss': The button action loss.
                  'entropy': The entropy of the action distribution.
                  'bc_weight': The weight used for the behavior cloning loss.
        :rtype: Dict[str, torch.Tensor]
        :raises AssertionError: If 'agent_action' is not in the batch.
        """
        assert 'agent_action' in batch, "key `agent_action` is required for behavior cloning."
        agent_action = batch['agent_action']
        pi_logits = latents['pi_logits']
        log_prob = mine_policy.pi_head.logprob(agent_action, pi_logits, return_dict=True)
        entropy  = mine_policy.pi_head.entropy(pi_logits, return_dict=True)
        camera_mask = (agent_action['camera'] != 60).float().squeeze(-1)
        global_mask = batch.get('mask', torch.ones_like(camera_mask))
        logp_camera = (log_prob['camera'] * global_mask * camera_mask).sum(-1)
        logp_buttons = (log_prob['buttons'] * global_mask).sum(-1)
        entropy_camera  = (entropy['camera'] * global_mask * camera_mask).sum(-1)
        entropy_buttons = (entropy['buttons'] * global_mask).sum(-1)
        camera_loss, button_loss = -logp_camera, -logp_buttons
        bc_loss = camera_loss + button_loss
        entropy = entropy_camera + entropy_buttons
        result = {
            'loss': bc_loss.mean() * self.weight,
            'camera_loss': camera_loss.mean(),
            'button_loss': button_loss.mean(),
            'entropy': entropy.mean(),
            'bc_weight': self.weight,
        }
        return result