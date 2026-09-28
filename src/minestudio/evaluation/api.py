"""Composition root for task-aware local evaluation and research artifacts."""

import logging
import platform
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, replace
from functools import partial
from typing import Any, Protocol, TypeVar
from uuid import uuid4

from minestudio import __version__
from minestudio.core import Info, MinecraftAction, Observation, StepResult
from minestudio.data.storage.trajectory import write_json
from minestudio.envs.protocols import Environment
from minestudio.evaluation.config import EvaluationConfig, EvaluationSuite
from minestudio.evaluation.results import EvaluationResult, summarize
from minestudio.policies import Policy
from minestudio.rollout import (
    Recorder,
    RolloutCase,
    RolloutConfig,
    RolloutRunner,
    TrajectoryRecorder,
)
from minestudio.tasks import TaskSpec

StateT = TypeVar("StateT")
logger = logging.getLogger(__name__)


class EvaluationEnvFactory(Protocol):
    def __call__(self, *, task: TaskSpec) -> Environment: ...


class _CheckedTaskEnv:
    """A suite's task must be applied by its factory, including resolved budgets."""

    def __init__(self, env: Environment, task: TaskSpec) -> None:
        self.env = env
        self.task = task

    def reset(
        self, *, seed: int | None = None, options: Info | None = None
    ) -> tuple[Observation, Info]:
        observation, info = self.env.reset(seed=seed, options=options)
        if info.get("task") != self.task.to_dict():
            raise ValueError("Evaluation env_factory did not apply the resolved TaskSpec")
        return observation, info

    def step(self, action: MinecraftAction) -> StepResult:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()


def _create_env(factory: EvaluationEnvFactory, task: TaskSpec) -> Environment:
    return _CheckedTaskEnv(factory(task=task), task)


def evaluate(
    policy_factory: Callable[[], Policy[StateT]],
    *,
    suite: EvaluationSuite,
    env_factory: EvaluationEnvFactory,
    config: EvaluationConfig,
    recorders: Sequence[Recorder] = (),
) -> EvaluationResult:
    """Evaluate exact task/seed cases and retain every retry with its own artifacts.

    Policy revisions are caller assertions, saved verbatim for audit. For a
    loaded VPT export, use its manifest hash as policy_revision. Custom policy
    RNGs remain caller-managed; deterministic=True is the evaluation default.
    """
    cases: list[RolloutCase] = []
    policy_identity: Info = {
        "id": config.policy_id,
        "revision": config.policy_revision,
        "processor_revision": config.processor_revision,
        "codec_revision": config.codec_revision,
    }
    budgets: list[int] = []
    for task in suite.tasks:
        budget = (
            config.max_episode_steps
            if config.max_episode_steps is not None
            else task.max_episode_steps
        )
        if budget is None:
            raise ValueError(f"Task {task.id} requires a finite evaluation step budget")
        resolved = replace(task, max_episode_steps=budget)
        budgets.append(budget)
        for seed in suite.seeds:
            for repeat in range(suite.repeats):
                metadata: Info = {
                    "suite": {"id": suite.id, "version": suite.version},
                    "task": resolved.to_dict(),
                    "budget_source": "evaluation_override"
                    if config.max_episode_steps is not None
                    else "task",
                    "original_task_budget": task.max_episode_steps,
                    "repeat": repeat,
                    "policy": policy_identity,
                    "deterministic": config.deterministic,
                    "policy_rng": "caller-managed",
                }
                cases.append(
                    RolloutCase(
                        len(cases), seed, partial(_create_env, env_factory, resolved), metadata
                    )
                )
    rollout_config = RolloutConfig(
        num_envs=config.num_envs,
        num_episodes=len(cases),
        max_episode_steps=max(budgets),
        max_retries=config.max_retries,
        deterministic=config.deterministic,
    )
    run_id = uuid4().hex
    report_path = (
        config.output_dir / run_id / "evaluation.json" if config.output_dir is not None else None
    )
    config_data = asdict(config)
    config_data["output_dir"] = str(config.output_dir) if config.output_dir is not None else None
    report: dict[str, Any] = {
        "format": "minestudio.evaluation",
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "minestudio_version": __version__,
        "python_version": platform.python_version(),
        "suite": {
            "id": suite.id,
            "version": suite.version,
            "tasks": [task.to_dict() for task in suite.tasks],
            "seeds": list(suite.seeds),
            "repeats": suite.repeats,
        },
        "config": config_data,
        "resolved_cases": [
            {"case_id": case.case_id, "seed": case.seed, "metadata": case.metadata}
            for case in cases
        ],
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=False)
        write_json(report_path, report)
    try:
        with ExitStack() as resources:
            active_recorders = list(recorders)
            if config.output_dir is not None:
                active_recorders.append(
                    resources.enter_context(TrajectoryRecorder(config.output_dir))
                )
                if config.record_video:
                    from minestudio.rollout.video import VideoRecorder

                    active_recorders.append(
                        resources.enter_context(VideoRecorder(config.output_dir))
                    )
            runner = resources.enter_context(
                RolloutRunner(
                    partial(_create_env, env_factory, suite.tasks[0]),
                    policy_factory,
                    rollout_config,
                    recorders=active_recorders,
                    cases=cases,
                )
            )
            rollout = runner.collect(run_id=run_id)
        result = summarize(rollout, requested=len(cases), report_path=report_path)
        if report_path is not None:
            payload: dict[str, Any] = asdict(result)
            payload["report_path"] = str(report_path)
            report.update(status="finished", result=payload)
            write_json(report_path, report)
    except BaseException as error:
        if report_path is not None:
            report.update(status="aborted", error=f"{type(error).__name__}: {error}")
            try:
                write_json(report_path, report)
            except Exception:
                logger.exception("Failed to save aborted evaluation report to %s", report_path)
        raise
    return result
