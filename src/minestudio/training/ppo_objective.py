"""GAE and clipped PPO objectives in explicit reward units."""

from dataclasses import dataclass

import torch

from minestudio.policies.batches import ActorCriticEvaluation, PolicyBatch
from minestudio.training.ppo_config import PPOConfig


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Last axis is time. Truncations bootstrap but stop traces; terminals do neither.

    next_values must refer to the final step observation, never an auto-reset frame.
    A rollout budget cut is a truncation when the environment resets for the next round.
    """
    if (
        rewards.ndim < 1
        or rewards.shape[-1] == 0
        or any(x.shape != rewards.shape for x in (values, next_values, terminated, truncated))
    ):
        raise ValueError("GAE requires matching nonempty [...,T] arrays")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("GAE boundary flags must be bool")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("GAE discounts must be in [0,1]")
    if any(not torch.isfinite(x).all() for x in (rewards, values, next_values)):
        raise ValueError("Non-finite GAE inputs")
    with torch.no_grad():
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        carry = torch.zeros_like(rewards[..., 0], dtype=torch.float32)
        for t in reversed(range(rewards.shape[-1])):
            delta = (
                rewards[..., t]
                + gamma * (~terminated[..., t]) * next_values[..., t]
                - values[..., t]
            )
            carry = delta + gamma * gae_lambda * (~(terminated[..., t] | truncated[..., t])) * carry
            advantages[..., t] = carry
        return advantages, advantages + values


@dataclass(frozen=True)
class PPOLoss:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def ppo_loss(
    evaluation: ActorCriticEvaluation,
    batch: PolicyBatch,
    *,
    old_log_prob: torch.Tensor,
    old_value: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
) -> PPOLoss:
    """Clipped surrogate plus 0.5*MSE value loss and entropy bonus, averaged over mask."""
    mask = batch.valid_mask
    tensors = (
        evaluation.log_prob,
        evaluation.value,
        evaluation.entropy,
        old_log_prob,
        old_value,
        advantages,
        returns,
    )
    if mask.dtype != torch.bool or not mask.any() or any(t.shape != mask.shape for t in tensors):
        raise ValueError("PPO statistics and a nonempty supervision mask must match [B,T]")
    log_prob, value, entropy, old_log_prob, old_value, advantages, returns = [
        t[mask].float() for t in tensors
    ]
    if any(
        not torch.isfinite(t).all()
        for t in (log_prob, value, entropy, old_log_prob, old_value, advantages, returns)
    ):
        raise FloatingPointError("Non-finite PPO statistics")
    log_ratio = log_prob - old_log_prob.detach()
    ratio = log_ratio.exp()
    policy_loss = -torch.minimum(
        ratio * advantages.detach(),
        ratio.clamp(1 - config.clip_range, 1 + config.clip_range) * advantages.detach(),
    ).mean()
    error = (value - returns.detach()).square()
    if config.value_clip_range is not None:
        clipped = old_value.detach() + (value - old_value.detach()).clamp(
            -config.value_clip_range, config.value_clip_range
        )
        error = torch.maximum(error, (clipped - returns.detach()).square())
    value_loss = 0.5 * error.mean()
    loss = (
        policy_loss
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * entropy.mean()
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite PPO objective")
    return PPOLoss(
        loss,
        policy_loss,
        value_loss,
        entropy.mean(),
        ((ratio - 1) - log_ratio).mean(),
        ((ratio - 1).abs() > config.clip_range).float().mean(),
    )
