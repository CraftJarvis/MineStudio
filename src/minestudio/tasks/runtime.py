"""Per-environment task state and explicit initialization context."""

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, SupportsFloat, cast, runtime_checkable

import numpy as np

from minestudio.core import Info, MineStudioError, Observation
from minestudio.tasks.rules import EndRule
from minestudio.tasks.spec import TaskSpec


class TaskMetricError(MineStudioError, ValueError):
    """A configured metric is absent or cannot be interpreted as a finite number."""


@runtime_checkable
class TaskContext(Protocol):
    def execute_command(self, command: str) -> tuple[Observation, Info]: ...


def _metric(info: Info, path: tuple[str, ...], missing: str) -> float:
    node: object = info
    for index, component in enumerate(path):
        if not isinstance(node, Mapping) or component not in node:
            if missing == "zero" and index == len(path) - 1 and isinstance(node, Mapping):
                return 0.0
            raise TaskMetricError(f"Missing task metric: {'/'.join(path)}")
        node = cast(Mapping[str, object], node)[component]
    if isinstance(node, np.ndarray):
        array = cast("np.ndarray[Any, Any]", node)
        if array.shape != () or array.dtype.kind not in "iuf":
            raise TaskMetricError(f"Task metric must be scalar: {'/'.join(path)}")
        node = array.item()
    if isinstance(node, (bool, np.bool_)) or not isinstance(
        node, (int, float, np.integer, np.floating)
    ):
        raise TaskMetricError(f"Task metric is not numeric: {'/'.join(path)}")
    value = float(cast(SupportsFloat, node))
    if not math.isfinite(value):
        raise TaskMetricError(f"Task metric is not finite: {'/'.join(path)}")
    return value


@dataclass(frozen=True)
class TaskStep:
    reward: float
    terminated: bool
    truncated: bool
    info: Info


class TaskRuntime:
    """No rule or state is shared between episodes; failure wins simultaneous ends."""

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
        self._previous: list[float] = []
        self._baselines: list[float] = []
        self._steps = 0
        self._active = False

    def initialize(
        self, context: object, observation: Observation, info: Info
    ) -> tuple[Observation, Info]:
        self._active = False
        if self.spec.initialization:
            if not isinstance(context, TaskContext):
                raise TypeError("Task initialization requires an execute_command backend")
            for rule in self.spec.initialization:
                observation, info = context.execute_command(rule.command)
        self._previous = [_metric(info, r.metric, r.missing) for r in self.spec.rewards]
        self._baselines = [
            _metric(info, r.metric, r.missing) for r in (*self.spec.success, *self.spec.failure)
        ]
        self._steps = 0
        self._active = True
        return observation, {
            **info,
            "task": self.spec.to_dict(),
            "success": False if self.spec.success else None,
            "initialization_steps": len(self.spec.initialization),
        }

    def step(
        self, info: Info, backend_reward: float, terminated: bool, truncated: bool
    ) -> TaskStep:
        if not self._active:
            raise RuntimeError("Initialize task state before stepping")
        current = [_metric(info, r.metric, r.missing) for r in self.spec.rewards]
        contributions: dict[str, float] = {}
        for rule, before, after in zip(self.spec.rewards, self._previous, current, strict=True):
            delta = after - before
            contributions[rule.id] = rule.reward_per_unit * (
                max(delta, 0) if rule.mode == "increase" else delta
            )
        rules = (*self.spec.success, *self.spec.failure)
        matches = [
            self._matches(rule, baseline, info)
            for rule, baseline in zip(rules, self._baselines, strict=True)
        ]
        successful = any(matches[: len(self.spec.success)])
        failed = any(matches[len(self.spec.success) :])
        self._steps += 1
        reason = info.get(
            "end_reason",
            "backend_terminal" if terminated else "backend_truncation" if truncated else None,
        )
        if failed or successful:
            terminated = True
            # The task terminal state is known; an independent backend truncation may coexist.
            reason = "task_failure" if failed else "task_success"
        elif (
            self.spec.max_episode_steps is not None
            and self._steps >= self.spec.max_episode_steps
            and not (terminated or truncated)
        ):
            truncated = True
            reason = "task_time_limit"
        self._previous = current
        self._active = not (terminated or truncated)
        task_reward = sum(contributions.values())
        reward = backend_reward + task_reward
        if not math.isfinite(reward):
            raise TaskMetricError("Task reward is not finite")
        result_info: Info = {
            **info,
            "task": self.spec.to_dict(),
            "success": bool(successful and not failed) if self.spec.success else None,
            "end_reason": reason,
            "task_step": self._steps,
            "reward_components": {
                "backend": backend_reward,
                "task": task_reward,
                "rules": contributions,
            },
        }
        if truncated and reason == "task_time_limit":
            result_info["truncation_reason"] = reason
        return TaskStep(reward, terminated, truncated, result_info)

    @staticmethod
    def _matches(rule: EndRule, baseline: float, info: Info) -> bool:
        value = _metric(info, rule.metric, rule.missing)
        if rule.relative:
            value -= baseline
        compare = {
            "ge": operator.ge,
            "le": operator.le,
            "gt": operator.gt,
            "lt": operator.lt,
            "eq": operator.eq,
        }[rule.operator]
        return compare(value, rule.threshold)
