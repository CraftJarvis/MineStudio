import json

import numpy as np
import pytest

from minestudio.actions import noop_action
from minestudio.core import BackendError, ImageSize
from minestudio.data.storage import TrajectoryReader
from minestudio.envs import EnvConfig, MinecraftEnv
from minestudio.evaluation import EvaluationConfig, EvaluationSuite, evaluate
from minestudio.rollout import RolloutRunner, TrajectoryRecorder
from minestudio.tasks import EndRule, TaskSpec


class Backend:
    def __init__(self, config, *, fail=False):
        self.config = config
        self.fail = fail
        self.closed = False
        self.steps = 0

    def observation(self):
        s = self.config.image_size
        return {"image": np.full((s.height, s.width, 3), self.steps, np.uint8)}

    def reset(self, *, seed=None, options=None):
        self.seed = seed
        self.steps = 0
        return self.observation(), {"metrics": {"count": 0}}

    def step(self, action):
        if self.fail:
            raise BackendError("engine failed")
        self.steps += 1
        return self.observation(), 1.0, False, False, {"metrics": {"count": self.steps}}

    def close(self):
        self.closed = True


class Policy:
    def initial_state(self):
        return 0

    def act(self, observation, state=None, *, deterministic=False):
        return noop_action(), state + 1


def suite():
    return EvaluationSuite(
        "collect",
        "1",
        (
            TaskSpec(
                "collect",
                "1",
                "Collect",
                success=(EndRule(("metrics", "count"), 2),),
                max_episode_steps=3,
            ),
        ),
        (42, 43),
    )


def config(**kwargs):
    return EvaluationConfig("test-policy", "seed-0", "rgb-v1", "native-v1", **kwargs)


def test_exact_case_seeds_retries_and_artifacts(tmp_path):
    backends = []

    def factory(*, task):
        cfg = EnvConfig(image_size=ImageSize(4, 6))
        backend = Backend(cfg, fail=not backends)
        backends.append(backend)
        return MinecraftEnv(cfg, task=task, backend_factory=lambda _: backend)

    result = evaluate(
        Policy,
        suite=suite(),
        env_factory=factory,
        config=config(output_dir=tmp_path, num_envs=2, max_retries=1),
    )
    assert (result.requested, result.completed, result.failed, result.successes) == (2, 2, 0, 2)
    assert result.success_rate == 1 and len(result.attempts) == 3
    assert [b.seed for b in backends] == [42, 43, 42]
    assert all(b.closed for b in backends)
    assert result.mean_return == 2
    report = json.loads(result.report_path.read_text())
    assert report["config"]["policy_revision"] == "seed-0"
    failed = result.attempts[0]
    record = TrajectoryReader(tmp_path / result.run_id / failed.context.attempt_id).read()
    assert record.status == "failed" and record.transitions == ()
    completed = result.cases[0]
    trajectory = TrajectoryReader(tmp_path / result.run_id / completed.context.attempt_id).read()
    assert trajectory.as_episode().transitions[-1].info["success"] is True
    assert trajectory.metadata["context"]["metadata"]["task"]["id"] == "collect"


def test_evaluation_budget_override_is_applied_and_recorded():
    tasks = []

    def factory(*, task):
        tasks.append(task)
        return MinecraftEnv(task=task, backend_factory=Backend)

    result = evaluate(
        Policy, suite=suite(), env_factory=factory, config=config(max_episode_steps=1)
    )
    assert result.completed == 2 and result.success_rate == 0
    assert all(task.max_episode_steps == 1 for task in tasks)
    assert all(case.end_reason == "task_time_limit" for case in result.cases)
    assert result.cases[0].context.metadata["budget_source"] == "evaluation_override"
    assert result.cases[0].context.metadata["original_task_budget"] == 3


def test_no_completed_cases_has_undefined_success_rate():
    def factory(*, task):
        return MinecraftEnv(task=task, backend_factory=lambda cfg: Backend(cfg, fail=True))

    result = evaluate(Policy, suite=suite(), env_factory=factory, config=config(max_retries=1))
    assert result.failed == 2 and result.completed == 0
    assert result.success_rate is None and result.mean_return is None
    assert len(result.attempts) == 4


def test_factory_must_apply_the_task(tmp_path):
    def factory(*, task):
        return MinecraftEnv(backend_factory=Backend)

    with pytest.raises(ValueError, match="did not apply"):
        evaluate(Policy, suite=suite(), env_factory=factory, config=config(output_dir=tmp_path))
    report = json.loads(next(tmp_path.glob("*/evaluation.json")).read_text())
    assert report["status"] == "aborted"
    assert len(report["resolved_cases"]) == 2
    assert report["resolved_cases"][0]["seed"] == 42


def test_aborted_policy_preserves_partial_attempt(tmp_path):
    class BrokenPolicy(Policy):
        def act(self, observation, state=None, **kwargs):
            if state == 1:
                raise ValueError("model broke")
            return super().act(observation, state, **kwargs)

    with (
        TrajectoryRecorder(tmp_path) as recorder,
        RolloutRunner(lambda: Backend(EnvConfig()), BrokenPolicy, recorders=[recorder]) as runner,
        pytest.raises(ValueError, match="model broke"),
    ):
        runner.collect()
    path = next(iter(recorder.paths.values()))
    record = TrajectoryReader(path).read()
    assert record.status == "failed"
    assert len(record.transitions) == 1
    assert not record.transitions[0].terminated
    assert record.outcome["error_type"] == "ValueError"


def test_report_write_failure_preserves_original_error(tmp_path, monkeypatch, caplog):
    from minestudio.evaluation import api

    original_write = api.write_json

    def write_report(path, report):
        if report["status"] == "aborted":
            raise OSError("disk full")
        original_write(path, report)

    def factory(*, task):
        raise ValueError("invalid environment")

    monkeypatch.setattr(api, "write_json", write_report)
    with pytest.raises(ValueError, match="invalid environment"):
        evaluate(Policy, suite=suite(), env_factory=factory, config=config(output_dir=tmp_path))
    assert "disk full" in caplog.text


def test_recorder_closes_all_writers_when_one_fails(tmp_path, monkeypatch):
    from minestudio.data.storage import TrajectoryWriter
    from minestudio.rollout import EpisodeContext

    recorder = TrajectoryRecorder(tmp_path)
    for attempt in ("first", "second"):
        recorder.on_episode_start(EpisodeContext("run", 0, attempt, 42))
    closed = []
    original_close = TrajectoryWriter.close

    def close_writer(writer):
        original_close(writer)
        closed.append(writer.directory.name)
        if writer.directory.name == "first":
            raise OSError("manifest unavailable")

    monkeypatch.setattr(TrajectoryWriter, "close", close_writer)
    with pytest.raises(OSError, match="manifest unavailable"):
        recorder.close()
    assert closed == ["first", "second"]
    assert not recorder._writers
