"""Case identity, attempt outcomes and recorder lifecycle."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from minestudio.core import Info, Observation, Transition
from minestudio.envs.protocols import Environment


@dataclass(frozen=True)
class RolloutCase:
    """An exact environment seed and optional case-specific factory and metadata."""

    case_id: int
    seed: int
    env_factory: Callable[[], Environment] | None = None
    metadata: Info = field(default_factory=lambda: dict[str, object]())

    def __post_init__(self) -> None:
        if (
            type(self.case_id) is not int
            or self.case_id < 0
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError("case_id and seed must be nonnegative integers")


@dataclass(frozen=True)
class EpisodeContext:
    run_id: str
    case_id: int
    attempt_id: str
    seed: int
    metadata: Info = field(default_factory=lambda: dict[str, object]())


@dataclass(frozen=True)
class EpisodeResult:
    context: EpisodeContext
    status: Literal["completed", "failed", "cancelled"]
    num_steps: int
    total_reward: float
    terminated: bool = False
    truncated: bool = False
    error: str | None = None
    success: bool | None = None
    end_reason: str | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class RolloutResult:
    run_id: str
    attempts: tuple[EpisodeResult, ...]
    completed: int
    failed: int
    cancelled: int


class Recorder(Protocol):
    """Start/end bracket normal and retryable attempts, including failed resets.
    A fatal callback/programming exception aborts collection and may interrupt
    the recorder lifecycle; resource cleanup still runs.

    Observations and transitions are snapshots. Recorders may retain them; they
    must not mutate them. Exceptions raised by a recorder abort the run.
    """

    def on_episode_start(self, context: EpisodeContext) -> None: ...
    def on_reset(self, context: EpisodeContext, observation: Observation, info: Info) -> None: ...
    def on_transition(self, context: EpisodeContext, transition: Transition) -> None: ...
    def on_episode_end(self, result: EpisodeResult) -> None: ...
