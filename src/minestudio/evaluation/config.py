"""Versioned suites and the explicitly supplied identity of the evaluated policy."""

from dataclasses import dataclass
from pathlib import Path

from minestudio.tasks import TaskSpec


@dataclass(frozen=True)
class EvaluationSuite:
    id: str
    version: str
    tasks: tuple[TaskSpec, ...]
    seeds: tuple[int, ...]
    repeats: int = 1

    def __post_init__(self) -> None:
        if not self.id or not self.version:
            raise ValueError("Evaluation suite id/version are required")
        if (
            not isinstance(self.tasks, tuple)
            or not self.tasks
            or any(not isinstance(t, TaskSpec) or not t.success for t in self.tasks)
        ):
            raise ValueError("Evaluation tasks must declare deterministic success rules")
        if len({(t.id, t.version) for t in self.tasks}) != len(self.tasks):
            raise ValueError("Suite task identities must be unique")
        if (
            not isinstance(self.seeds, tuple)
            or not self.seeds
            or any(type(seed) is not int or seed < 0 for seed in self.seeds)
        ):
            raise ValueError("seeds must be a nonempty tuple of nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Duplicate seeds are ambiguous; use repeats")
        if type(self.repeats) is not int or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")


@dataclass(frozen=True)
class EvaluationConfig:
    policy_id: str
    policy_revision: str
    processor_revision: str
    codec_revision: str
    output_dir: Path | None = None
    num_envs: int = 1
    max_retries: int = 0
    max_episode_steps: int | None = None
    deterministic: bool = True
    record_video: bool = False

    def __post_init__(self) -> None:
        for value in (
            self.policy_id,
            self.policy_revision,
            self.processor_revision,
            self.codec_revision,
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Policy, processor and codec identities must be explicit")
        if (
            type(self.num_envs) is not int
            or self.num_envs < 1
            or type(self.max_retries) is not int
            or self.max_retries < 0
        ):
            raise ValueError("Invalid environment or retry count")
        if self.max_episode_steps is not None and (
            type(self.max_episode_steps) is not int or self.max_episode_steps < 1
        ):
            raise ValueError("Evaluation step override must be a positive integer")
        if self.record_video and self.output_dir is None:
            raise ValueError("Video recording requires output_dir")
        if type(self.deterministic) is not bool or type(self.record_video) is not bool:
            raise ValueError("deterministic and record_video must be booleans")
