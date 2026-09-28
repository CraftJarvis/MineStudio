"""Local rollout budgets count cases globally, including when retried."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutConfig:
    backend: str = "local"
    num_envs: int = 1
    num_episodes: int = 1
    max_episode_steps: int = 1200
    seed: int = 0
    deterministic: bool = False
    max_retries: int = 0

    def __post_init__(self) -> None:
        if self.backend != "local":
            raise ValueError("Only backend='local' is implemented in this alpha")
        for name in ("num_envs", "num_episodes", "max_episode_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "max_retries"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
