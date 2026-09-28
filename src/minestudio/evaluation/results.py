"""Case-level metrics: retries stay in the attempt log, not in the denominator."""

from dataclasses import dataclass
from pathlib import Path

from minestudio.rollout.records import EpisodeResult, RolloutResult


@dataclass(frozen=True)
class EvaluationResult:
    run_id: str
    requested: int
    completed: int
    failed: int
    cancelled: int
    successes: int
    success_rate: float | None
    mean_return: float | None
    mean_steps: float | None
    cases: tuple[EpisodeResult, ...]
    attempts: tuple[EpisodeResult, ...]
    report_path: Path | None = None


def summarize(
    result: RolloutResult, *, requested: int, report_path: Path | None = None
) -> EvaluationResult:
    final: dict[int, EpisodeResult] = {}
    for attempt in result.attempts:
        final[attempt.context.case_id] = attempt
    completed = [case for case in final.values() if case.status == "completed"]
    if any(case.success is None for case in completed):
        raise ValueError("Completed evaluation cases must report explicit task success")
    if (
        len(completed) != result.completed
        or result.completed + result.failed + result.cancelled != requested
    ):
        raise ValueError("Inconsistent rollout case counts")
    successes = sum(case.success is True for case in completed)
    return EvaluationResult(
        result.run_id,
        requested,
        result.completed,
        result.failed,
        result.cancelled,
        successes,
        successes / len(completed) if completed else None,
        sum(case.total_reward for case in completed) / len(completed) if completed else None,
        sum(case.num_steps for case in completed) / len(completed) if completed else None,
        tuple(final[key] for key in sorted(final)),
        result.attempts,
        report_path,
    )
