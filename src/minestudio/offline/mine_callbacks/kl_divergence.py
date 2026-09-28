'''
Date: 2024-12-12 13:10:58
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-05-27 14:14:37
FilePath: /MineStudio/minestudio/offline/mine_callbacks/kl_divergence.py
'''

import torch
from typing import Dict, Any
from minestudio.models import MinePolicy
from minestudio.offline.mine_callbacks.callback import ObjectiveCallback

class KLDivergenceCallback(ObjectiveCallback):
    """
    A callback to compute the KL divergence between two Gaussian distributions.

    This callback is typically used in Variational Autoencoders (VAEs) or similar
    models where a prior distribution is regularized towards a posterior distribution.
    The KL divergence is calculated between a posterior (q) and a prior (p) distribution,
    both assumed to be Gaussian and defined by their means (mu) and log variances (log_var).
    """
        
    def __init__(self, weight: float=1.0):
        """
        Initializes the KLDivergenceCallback.

        :param weight: The weight to apply to the KL divergence loss. Defaults to 1.0.
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
        Calculates the KL divergence loss.

        It retrieves the parameters (mu and log_var) of the posterior and prior
        distributions from the `latents` dictionary. Then, it computes the
        KL divergence using the `kl_divergence` method and returns it as part of
        the loss dictionary.

        :param batch: A dictionary containing the batch data.
        :type batch: Dict[str, Any]
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :param step_name: The name of the current step (e.g., 'train', 'val').
        :type step_name: str
        :param latents: A dictionary containing the policy's latent outputs.
                        Must include 'posterior_dist' and 'prior_dist',
                        each with 'mu' and 'log_var' keys.
        :type latents: Dict[str, torch.Tensor]
        :param mine_policy: The MinePolicy model.
        :type mine_policy: MinePolicy
        :returns: A dictionary containing the calculated losses and metrics:
                  'loss': The weighted KL divergence loss.
                  'kl_div': The mean KL divergence.
                  'kl_weight': The weight used for the KL divergence loss.
        :rtype: Dict[str, torch.Tensor]
        """
        posterior_dist = latents['posterior_dist']
        prior_dist = latents['prior_dist']
        
        q_mu, q_log_var = posterior_dist['mu'], posterior_dist['log_var']
        p_mu, p_log_var = prior_dist['mu'], prior_dist['log_var']
        
        kl_div = self.kl_divergence(q_mu, q_log_var, p_mu, p_log_var)
        result = {
            'loss': kl_div.mean() * self.weight,
            'kl_div': kl_div.mean(),
            'kl_weight': self.weight,
        }
        return result

    def kl_divergence(self, q_mu, q_log_var, p_mu, p_log_var):
        """
        Computes the KL divergence between two Gaussian distributions q and p.

        KL(q || p) = -0.5 * sum(1 + log(sigma_q^2 / sigma_p^2) - (sigma_q^2 / sigma_p^2) - ((mu_q - mu_p)^2 / sigma_p^2))
        where sigma^2 = exp(log_var).

        :param q_mu: Mean of the posterior distribution q. Shape: (B, D)
        :type q_mu: torch.Tensor
        :param q_log_var: Log variance of the posterior distribution q. Shape: (B, D)
        :type q_log_var: torch.Tensor
        :param p_mu: Mean of the prior distribution p. Shape: (B, D)
        :type p_mu: torch.Tensor
        :param p_log_var: Log variance of the prior distribution p. Shape: (B, D)
        :type p_log_var: torch.Tensor
        :returns: The KL divergence for each element in the batch. Shape: (B)
        :rtype: torch.Tensor
        """
        # shape: (B, D)
        KL = -0.5 * torch.sum(
            1 + (q_log_var - p_log_var) - (q_log_var - p_log_var).exp() - (q_mu - p_mu).pow(2) / p_log_var.exp(), dim=-1
        ) # shape: (B)
        return KL