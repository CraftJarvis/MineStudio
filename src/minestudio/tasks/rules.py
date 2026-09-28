"""Versioned, declarative task rules. No environment or training imports."""

import math
from dataclasses import dataclass
from typing import Literal


def finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def validate_metric(metric: tuple[str, ...], missing: str) -> None:
    if (
        not isinstance(metric, tuple)
        or not metric
        or any(not isinstance(p, str) or not p for p in metric)
    ):
        raise ValueError("metric must be a nonempty tuple of nonempty path components")
    if missing not in ("error", "zero"):
        raise ValueError("missing must be 'error' or 'zero'")


@dataclass(frozen=True)
class InitializationRule:
    """Execute one Minecraft command, in order, before the initial observation."""

    command: str
    kind: Literal["command"] = "command"
    version: int = 1

    def __post_init__(self) -> None:
        if self.kind != "command" or type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported initialization rule kind/version")
        if (
            not isinstance(self.command, str)
            or not self.command.startswith("/")
            or any(c in self.command for c in "\n\r\x00")
        ):
            raise ValueError("command must be one slash-prefixed Minecraft command")


@dataclass(frozen=True)
class RewardRule:
    """Reward the change in a metric from the previous observation.

    increase ignores negative changes; delta retains them. Initial metrics are
    baselined after initialization, so setup commands never award task reward.
    """

    id: str
    metric: tuple[str, ...]
    reward_per_unit: float = 1.0
    mode: Literal["increase", "delta"] = "increase"
    missing: Literal["error", "zero"] = "error"
    kind: Literal["metric_change"] = "metric_change"
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Reward rule id must be nonempty")
        if self.kind != "metric_change" or type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported reward rule kind/version")
        validate_metric(self.metric, self.missing)
        finite_number(self.reward_per_unit, "reward_per_unit")
        if self.mode not in ("increase", "delta"):
            raise ValueError("mode must be increase or delta")


@dataclass(frozen=True)
class EndRule:
    """Compare a metric's absolute value or change since reset to a threshold."""

    metric: tuple[str, ...]
    threshold: float
    operator: Literal["ge", "le", "gt", "lt", "eq"] = "ge"
    relative: bool = False
    missing: Literal["error", "zero"] = "error"
    kind: Literal["metric_threshold"] = "metric_threshold"
    version: int = 1

    def __post_init__(self) -> None:
        if self.kind != "metric_threshold" or type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported end rule kind/version")
        validate_metric(self.metric, self.missing)
        finite_number(self.threshold, "threshold")
        if self.operator not in ("ge", "le", "gt", "lt", "eq") or type(self.relative) is not bool:
            raise ValueError("Invalid end-rule operator or relative flag")
