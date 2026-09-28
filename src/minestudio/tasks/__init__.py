"""Declarative tasks and per-environment execution state."""

from importlib.resources import files

from minestudio.tasks.rules import EndRule, InitializationRule, RewardRule
from minestudio.tasks.runtime import TaskContext, TaskMetricError, TaskRuntime
from minestudio.tasks.spec import TaskSpec

__all__ = [
    "EndRule",
    "InitializationRule",
    "RewardRule",
    "TaskContext",
    "TaskMetricError",
    "TaskRuntime",
    "TaskSpec",
    "load_task",
]


def load_task(name: str) -> TaskSpec:
    """Load a packaged task by its exact id; no directory scanning or code registry."""
    if name not in {"collect_oak_log"}:
        raise ValueError(f"Unknown built-in task: {name}")
    import json

    return TaskSpec.from_dict(
        json.loads(
            (files("minestudio.tasks") / "configs" / f"{name}.json").read_text(encoding="utf-8")
        )
    )
