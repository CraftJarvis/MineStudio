"""Validated environment configuration; constructing it has no side effects."""

from dataclasses import dataclass, field

from minestudio.core import ImageSize


@dataclass(frozen=True)
class EnvConfig:
    image_size: ImageSize = field(default_factory=lambda: ImageSize(224, 224))
    render_size: ImageSize = field(default_factory=lambda: ImageSize(360, 640))
    max_episode_steps: int | None = None
    warmup_steps: int = 20

    def __post_init__(self) -> None:
        if self.max_episode_steps is not None and (
            type(self.max_episode_steps) is not int or self.max_episode_steps < 1
        ):
            raise ValueError("max_episode_steps must be a positive integer or None")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
