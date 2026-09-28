"""Environment/backend structural contracts, independent of Gymnasium imports."""

from typing import Protocol

from minestudio.core import Info, MinecraftAction, Observation, StepResult


class Environment(Protocol):
    def reset(
        self, *, seed: int | None = None, options: Info | None = None
    ) -> tuple[Observation, Info]: ...
    def step(self, action: MinecraftAction) -> StepResult: ...
    def close(self) -> None: ...


class EnvBackend(Environment, Protocol):
    """Owned backend. Return configured RGB images and raise on failed transitions."""
