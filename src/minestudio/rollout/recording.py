"""In-memory recording for small experiments and tests."""

from copy import deepcopy

from minestudio.core import Episode, Info, Observation, Transition
from minestudio.rollout.records import EpisodeContext, EpisodeResult


class MemoryRecorder:
    """Retain completed episodes and partial failed attempts. Images use RAM."""

    def __init__(self) -> None:
        self.initial_observations: dict[str, Observation] = {}
        self.transitions: dict[str, list[Transition]] = {}
        self.results: list[EpisodeResult] = []
        self.episodes: dict[str, Episode] = {}

    def on_episode_start(self, context: EpisodeContext) -> None:
        self.transitions[context.attempt_id] = []

    def on_reset(self, context: EpisodeContext, observation: Observation, info: Info) -> None:
        self.initial_observations[context.attempt_id] = deepcopy(observation)

    def on_transition(self, context: EpisodeContext, transition: Transition) -> None:
        self.transitions[context.attempt_id].append(deepcopy(transition))

    def on_episode_end(self, result: EpisodeResult) -> None:
        self.results.append(result)
        if result.status == "completed":
            key = result.context.attempt_id
            self.episodes[key] = Episode(
                self.initial_observations[key], tuple(self.transitions[key])
            )
