"""Immutable task configuration with strict JSON/YAML loading."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

from minestudio.core import MissingDependencyError
from minestudio.tasks.rules import EndRule, InitializationRule, RewardRule


@dataclass(frozen=True)
class TaskSpec:
    id: str
    version: str
    instruction: str
    initialization: tuple[InitializationRule, ...] = ()
    rewards: tuple[RewardRule, ...] = ()
    success: tuple[EndRule, ...] = ()
    failure: tuple[EndRule, ...] = ()
    max_episode_steps: int | None = None

    def __post_init__(self) -> None:
        for name in ("id", "version", "instruction"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Task {name} must be a nonempty string")
        for name, rule_type in (
            ("initialization", InitializationRule),
            ("rewards", RewardRule),
            ("success", EndRule),
            ("failure", EndRule),
        ):
            rules = getattr(self, name)
            if not isinstance(rules, tuple) or any(
                not isinstance(r, rule_type) for r in cast(tuple[object, ...], rules)
            ):
                raise TypeError(f"{name} must be a tuple of {rule_type.__name__}")
        if len({r.id for r in self.rewards}) != len(self.rewards):
            raise ValueError("Reward rule ids must be unique within a task")
        if self.max_episode_steps is not None and (
            type(self.max_episode_steps) is not int or self.max_episode_steps < 1
        ):
            raise ValueError("max_episode_steps must be a positive integer or None")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return cast(dict[str, Any], json.loads(json.dumps(asdict(self))))

    @classmethod
    def from_dict(cls, value: object) -> "TaskSpec":
        if not isinstance(value, dict) or any(
            not isinstance(k, str) for k in cast(dict[object, object], value)
        ):
            raise ValueError("Task configuration must be an object")
        data: dict[str, Any] = dict(cast(dict[str, Any], value))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown task fields: {sorted(unknown)}")
        for name, rule_type in (
            ("initialization", InitializationRule),
            ("rewards", RewardRule),
            ("success", EndRule),
            ("failure", EndRule),
        ):
            raw = data.get(name, [])
            if not isinstance(raw, list):
                raise ValueError(f"{name} must be a list")
            rules: list[Any] = []
            for item in cast(list[object], raw):
                if not isinstance(item, dict):
                    raise ValueError(f"Each {name} rule must be an object")
                rule_data: dict[str, Any] = dict(cast(dict[str, Any], item))
                if "metric" in rule_data:
                    if not isinstance(rule_data["metric"], list):
                        raise ValueError("metric must be a list of path components")
                    rule_data["metric"] = tuple(cast(list[Any], rule_data["metric"]))
                try:
                    rules.append(rule_type(**rule_data))
                except TypeError as error:
                    raise ValueError(f"Invalid {name} rule: {error}") from error
            data[name] = tuple(rules)
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError(f"Invalid task: {error}") from error

    @classmethod
    def from_json(cls, path: str | Path) -> "TaskSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskSpec":
        try:
            import yaml
        except ImportError as error:
            raise MissingDependencyError("Install minestudio[tasks] for YAML task files") from error
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
