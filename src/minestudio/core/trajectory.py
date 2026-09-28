"""Online records; failures do not create synthetic transitions."""

from dataclasses import dataclass, field

import numpy as np

from minestudio.core.types import Info, MinecraftAction, Observation


@dataclass(frozen=True)
class Transition:
    """One valid step, including its final observation."""

    observation: Observation
    action: MinecraftAction
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool
    info: Info = field(default_factory=lambda: dict[str, object]())


@dataclass(frozen=True)
class Episode:
    """A completed online episode with T transitions and T+1 observations.

    Historical unknown rewards or boundaries require a masked representation;
    they must not be coerced into this fully observed type.
    """

    initial_observation: Observation
    transitions: tuple[Transition, ...]

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("A completed episode needs at least one transition")
        previous = self.initial_observation
        for transition in self.transitions:
            if not np.array_equal(previous["image"], transition.observation["image"]):
                raise ValueError("Episode observations are not temporally aligned")
            previous = transition.next_observation
        if any(t.terminated or t.truncated for t in self.transitions[:-1]):
            raise ValueError("An episode cannot contain transitions after its boundary")
        if not (self.transitions[-1].terminated or self.transitions[-1].truncated):
            raise ValueError("A completed episode must end at a known boundary")

    @property
    def observations(self) -> tuple[Observation, ...]:
        return (self.initial_observation, *(t.next_observation for t in self.transitions))
