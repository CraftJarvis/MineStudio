"""Policy inference requires no environment or training framework."""

from typing import Protocol, TypeVar, runtime_checkable

from minestudio.core import MinecraftAction, Observation

StateT = TypeVar("StateT")


@runtime_checkable
class Policy(Protocol[StateT]):
    """Caller-owned state for one stream; act returns a native action."""

    def initial_state(self) -> StateT: ...
    def act(
        self,
        observation: Observation,
        state: StateT | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[MinecraftAction, StateT]: ...
