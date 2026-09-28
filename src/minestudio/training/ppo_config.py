"""Explicit budgets and numerical semantics for synchronous recurrent PPO."""

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PPOConfig:
    """FP32, constant LR, frozen value normalization and round-boundary resets.

    Each round has num_envs * rollout_steps fresh on-policy transitions. Every
    environment resets when weights change. Training sequences retain detached
    behavior-policy caches: a declared truncated-BPTT approximation within a round.
    environment_id must identify the caller's task/environment configuration.
    """

    output_dir: str | Path
    environment_id: str
    iterations: int = 2
    num_envs: int = 1
    rollout_steps: int = 256
    sequence_length: int = 32
    minibatch_sequences: int = 4
    update_epochs: int = 2
    max_episode_steps: int = 1200
    learning_rate: float = 1e-6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float | None = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 1.0
    normalize_advantage: bool = True
    target_kl: float | None = 0.02
    behavior_tolerance: float = 5e-4
    seed: int = 13
    env_seed: int = 42
    save_rollouts: bool = False

    def __post_init__(self) -> None:
        for key in (
            "iterations",
            "num_envs",
            "rollout_steps",
            "sequence_length",
            "minibatch_sequences",
            "update_epochs",
            "max_episode_steps",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.rollout_steps % self.sequence_length:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        if not isinstance(self.environment_id, str) or not self.environment_id.strip():
            raise ValueError("environment_id must identify a fixed task/environment configuration")
        for key in ("seed", "env_seed"):
            if type(getattr(self, key)) is not int or not 0 <= getattr(self, key) < 2**32:
                raise ValueError(f"{key} must be an unsigned 32-bit integer")
        for key in ("learning_rate", "max_grad_norm", "behavior_tolerance"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("gamma", "gae_lambda"):
            if not math.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        if not math.isfinite(self.clip_range) or not 0 < self.clip_range < 1:
            raise ValueError("clip_range must be in (0,1)")
        for key in ("value_coefficient", "entropy_coefficient"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        for key in ("value_clip_range", "target_kl"):
            value = getattr(self, key)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be positive and finite or None")
        for key in ("normalize_advantage", "save_rollouts"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
