"""Synchronous on-policy sequences with recorded behavior probabilities and caches."""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from minestudio.core.types import snapshot_observation
from minestudio.envs.protocols import Environment
from minestudio.policies.batches import PolicyBatch
from minestudio.training._checkpoint import tensor_tree
from minestudio.training.ppo_config import PPOConfig
from minestudio.training.ppo_objective import compute_gae


def batch_to(batch: PolicyBatch, device: str | torch.device) -> PolicyBatch:
    return PolicyBatch(
        {k: v.to(device) for k, v in batch.inputs.items()},
        {k: v.to(device) for k, v in batch.actions.items()},
        batch.valid_mask.to(device),
        batch.first_mask.to(device),
    )


@dataclass(frozen=True)
class PPOSequence:
    batch: PolicyBatch
    initial_state: Any
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    rewards: torch.Tensor
    next_values: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    episode_ids: torch.Tensor
    policy_version: int
    state_version: int


@dataclass(frozen=True)
class PPORollout:
    sequences: tuple[PPOSequence, ...]
    episodes: tuple[dict[str, Any], ...]
    next_episode: int
    policy_version: int

    def payload(self) -> dict[str, Any]:
        """Plain tensor dictionaries loadable with torch.load(weights_only=True)."""
        return {"format": "minestudio.ppo-rollout", "schema_version": 1, **asdict(self)}


def collect_ppo_rollout(
    policy: Any,
    environments: Sequence[Environment],
    *,
    config: PPOConfig,
    policy_version: int,
    next_episode: int = 0,
) -> PPORollout:
    """Reset each environment for a fresh behavior-policy round; never auto-retry failures.

    Encoded samples are retained directly. Per-sequence caches are detached snapshots
    of this round's behavior policy, not caches reconstructed from native actions.
    All remaining streams are explicitly truncated at the round's sampling budget.
    """
    if policy.training or len(environments) != config.num_envs:
        raise ValueError("PPO sampling requires eval mode and the configured environment count")
    sequences, episodes = [], []
    for slot, env in enumerate(environments):

        def reset(env=env):
            nonlocal next_episode
            episode_id = next_episode
            seed = (config.env_seed + episode_id) % 2**32
            next_episode += 1
            obs, _ = env.reset(seed=seed)
            return snapshot_observation(obs), policy.initial_state(), episode_id, seed

        observation, state, episode_id, seed = reset()
        first, episode_steps, episode_return = True, 0, 0.0
        rewards, values, log_probs, next_values = [], [], [], []
        terminated_flags, truncated_flags, episode_ids = [], [], []
        chunks, chunk_observations, codes, first_flags = [], [], [], []
        for t in range(config.rollout_steps):
            if t % config.sequence_length == 0:
                initial_state = policy.batch_state(state, device="cpu")
            chunk_observations.append(observation)
            first_flags.append(first)
            step, following_state = policy.act_with_stats(observation, state, deterministic=False)
            if not math.isfinite(step.log_prob) or not math.isfinite(step.value):
                raise FloatingPointError("Non-finite behavior policy statistics")
            codes.append(dict(step.policy_action))
            following, reward, terminated, truncated, _ = env.step(step.action)
            following = snapshot_observation(following)
            if (
                not math.isfinite(reward)
                or type(terminated) is not bool
                or type(truncated) is not bool
            ):
                raise ValueError("Invalid environment reward or boundary flags")
            episode_steps += 1
            episode_return += reward
            round_cut = t == config.rollout_steps - 1 and not (terminated or truncated)
            limit_cut = episode_steps >= config.max_episode_steps and not (terminated or truncated)
            truncated = truncated or round_cut or limit_cut
            rewards.append(float(reward))
            values.append(step.value)
            log_probs.append(step.log_prob)
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
            episode_ids.append(episode_id)
            # Reuse the next sampled V(s) within episodes; only boundaries need an extra forward.
            next_values.append(
                0.0
                if terminated
                else policy.value(following, following_state)
                if truncated
                else None
            )
            if (t + 1) % config.sequence_length == 0:
                batch = policy.prepare_sequence(chunk_observations, codes, first_flags)
                chunks.append((batch_to(batch, "cpu"), initial_state))
                chunk_observations, codes, first_flags = [], [], []
            if terminated or truncated:
                episodes.append(
                    {
                        "slot": slot,
                        "episode_id": episode_id,
                        "seed": seed,
                        "steps": episode_steps,
                        "return": episode_return,
                        "terminated": terminated,
                        "truncated": truncated,
                        "cut": "rollout"
                        if round_cut
                        else "episode_limit"
                        if limit_cut
                        else "environment",
                    }
                )
                if t + 1 < config.rollout_steps:
                    observation, state, episode_id, seed = reset()
                    first, episode_steps, episode_return = True, 0, 0.0
            else:
                observation, state, first = following, following_state, False
        for t in range(config.rollout_steps - 1):
            if next_values[t] is None:
                next_values[t] = values[t + 1]
        tensors = {
            "rewards": torch.tensor([rewards]),
            "old_value": torch.tensor([values]),
            "old_log_prob": torch.tensor([log_probs]),
            "next_values": torch.tensor([next_values]),
            "terminated": torch.tensor([terminated_flags], dtype=torch.bool),
            "truncated": torch.tensor([truncated_flags], dtype=torch.bool),
            "episode_ids": torch.tensor([episode_ids], dtype=torch.long),
        }
        advantages, returns = compute_gae(
            tensors["rewards"],
            tensors["old_value"],
            tensors["next_values"],
            tensors["terminated"],
            tensors["truncated"],
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        tensors.update(advantages=advantages, returns=returns)
        for chunk, (batch, initial_state) in enumerate(chunks):
            start = chunk * config.sequence_length
            sequence = {
                k: v[:, start : start + config.sequence_length].clone() for k, v in tensors.items()
            }
            sequences.append(
                PPOSequence(
                    batch=batch,
                    initial_state=tensor_tree(initial_state, "cpu"),
                    policy_version=policy_version,
                    state_version=policy_version,
                    **sequence,
                )
            )
    if config.normalize_advantage:
        flat = torch.cat([s.advantages[s.batch.valid_mask] for s in sequences])
        mean, std = flat.mean(), flat.std(unbiased=False)
        sequences = [replace(s, advantages=(s.advantages - mean) / (std + 1e-8)) for s in sequences]
    return PPORollout(tuple(sequences), tuple(episodes), next_episode, policy_version)
