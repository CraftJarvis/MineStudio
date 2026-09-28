"""Offline windows preserve unknown observations, rewards and boundaries."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from minestudio.core.types import MinecraftAction


@dataclass(frozen=True)
class TrajectoryWindow:
    """T actions and T+1 RGB observations; zero-filled slots require false masks.

    episode_id identifies the source record, which need not be a full episode.
    first_mask marks known episode starts, never arbitrary window starts.
    """

    episode_id: str
    start_step: int
    observations: dict[str, NDArray[np.uint8]]
    actions: tuple[MinecraftAction, ...]
    rewards: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    valid_mask: NDArray[np.bool_]
    observation_mask: dict[str, NDArray[np.bool_]]
    action_mask: NDArray[np.bool_]
    reward_mask: NDArray[np.bool_]
    boundary_mask: NDArray[np.bool_]
    first_mask: NDArray[np.bool_]
    timebase: Literal["environment_steps", "legacy_rows"] = "environment_steps"

    def __post_init__(self) -> None:
        if self.timebase not in ("environment_steps", "legacy_rows"):
            raise ValueError("Unknown window timebase")
        length = len(self.actions)
        if not self.episode_id or self.start_step < 0 or length == 0:
            raise ValueError("A window needs an identity, nonnegative start and actions")
        for mask in (
            self.terminated,
            self.truncated,
            self.valid_mask,
            self.action_mask,
            self.reward_mask,
            self.boundary_mask,
            self.first_mask,
        ):
            if mask.dtype != np.bool_ or mask.shape != (length,):
                raise ValueError("Transition masks and flags must be bool[T]")
        for mask in (self.action_mask, self.reward_mask, self.boundary_mask, self.first_mask):
            if np.any(mask & ~self.valid_mask):
                raise ValueError("Padding cannot carry known labels or episode starts")
        if self.rewards.shape != (length,) or not np.isfinite(self.rewards).all():
            raise ValueError("Rewards must be finite [T] placeholders with reward_mask")
        image = self.observations["image"]
        if image.dtype != np.uint8 or image.ndim != 4 or image.shape[0] != length + 1:
            raise ValueError("Images must be uint8[T+1,H,W,3]")
        if image.shape[-1] != 3 or min(image.shape[1:3]) < 1:
            raise ValueError("Images must be nonempty RGB")
        mask = self.observation_mask["image"]
        if mask.dtype != np.bool_ or mask.shape != (length + 1,):
            raise ValueError("Image mask must be bool[T+1]")
