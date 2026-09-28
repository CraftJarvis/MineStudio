import json
from dataclasses import replace

import numpy as np
import pytest

from minestudio.actions import noop_action
from minestudio.core import BackendError, EpisodeStateError, ImageSize
from minestudio.envs import EnvConfig, MinecraftEnv
from minestudio.tasks import (
    EndRule,
    InitializationRule,
    RewardRule,
    TaskMetricError,
    TaskSpec,
    load_task,
)


class TaskBackend:
    def __init__(self, config):
        self.config = config
        self.quantity = 5
        self.health = 10
        self.commands = []
        self.closed = False

    def observation(self):
        size = self.config.image_size
        return {"image": np.full((size.height, size.width, 3), self.quantity, np.uint8)}

    def info(self):
        return {"metrics": {"quantity": self.quantity, "health": self.health}}

    def reset(self, *, seed=None, options=None):
        self.quantity, self.health = 5, 10
        return self.observation(), self.info()

    def execute_command(self, command):
        self.commands.append(command)
        if command == "/fail":
            raise BackendError("command failed")
        self.quantity += 1
        return self.observation(), self.info()

    def step(self, action):
        self.quantity += 1
        self.health -= 1
        return self.observation(), 2.0, False, False, self.info()

    def close(self):
        self.closed = True


def task(**kwargs):
    return TaskSpec(
        "collect",
        "1",
        "Collect two more items",
        rewards=(RewardRule("items", ("metrics", "quantity")),),
        success=(EndRule(("metrics", "quantity"), 2, relative=True),),
        **kwargs,
    )


def test_initialization_order_baseline_and_additive_reward():
    spec = task(initialization=(InitializationRule("/first"), InitializationRule("/second")))
    config = EnvConfig(image_size=ImageSize(4, 6))
    backend = TaskBackend(config)
    with MinecraftEnv(config, task=spec, backend_factory=lambda _: backend) as env:
        initial, info = env.reset(seed=9)
        assert backend.commands == ["/first", "/second"]
        assert initial["image"].max() == 7
        assert info["initialization_steps"] == 2
        assert info["task"] == spec.to_dict()
        first = env.step(noop_action())
        final = env.step(noop_action())
        assert first[1:4] == (3.0, False, False)
        assert final[1:4] == (3.0, True, False)
        assert final[4]["success"] is True
        assert final[4]["end_reason"] == "task_success"
        assert final[4]["reward_components"] == {
            "backend": 2.0,
            "task": 1.0,
            "rules": {"items": 1.0},
        }
        with pytest.raises(EpisodeStateError):
            env.step(noop_action())
        env.reset(seed=9)
        assert env.step(noop_action())[4]["success"] is False
    assert backend.closed


def test_failure_wins_simultaneous_success_and_budget():
    spec = task(failure=(EndRule(("metrics", "health"), 8, operator="le"),), max_episode_steps=2)
    with MinecraftEnv(task=spec, backend_factory=TaskBackend) as env:
        env.reset()
        env.step(noop_action())
        _, _, terminated, truncated, info = env.step(noop_action())
        assert (terminated, truncated, info["success"]) == (True, False, False)
        assert info["end_reason"] == "task_failure"


def test_task_budget_truncates_and_state_is_instance_local():
    spec = task(max_episode_steps=1)
    with (
        MinecraftEnv(task=spec, backend_factory=TaskBackend) as a,
        MinecraftEnv(task=spec, backend_factory=TaskBackend) as b,
    ):
        a.reset()
        b.reset()
        final_a = a.step(noop_action())
        final_b = b.step(noop_action())
        assert final_a[2:4] == final_b[2:4] == (False, True)
        assert final_b[4]["task_step"] == 1
        assert final_b[4]["end_reason"] == "task_time_limit"


def test_missing_telemetry_is_not_treated_as_zero():
    spec = task()
    spec = replace(spec, rewards=(RewardRule("missing", ("absent", "count"), missing="zero"),))
    backend = TaskBackend(EnvConfig())
    with (
        MinecraftEnv(task=spec, backend_factory=lambda _: backend) as env,
        pytest.raises(TaskMetricError, match="Missing"),
    ):
        env.reset()
    assert backend.closed


def test_failed_initialization_closes_backend():
    backend = TaskBackend(EnvConfig())
    with (
        MinecraftEnv(
            task=task(initialization=(InitializationRule("/fail"),)),
            backend_factory=lambda _: backend,
        ) as env,
        pytest.raises(BackendError, match="command failed"),
    ):
        env.reset()
    assert backend.closed


def test_task_loading_is_strict_and_roundtrips(tmp_path):
    spec = load_task("collect_oak_log")
    assert TaskSpec.from_dict(spec.to_dict()) == spec
    path = tmp_path / "task.json"
    path.write_text(json.dumps(spec.to_dict()))
    assert TaskSpec.from_json(path) == spec
    data = spec.to_dict()
    data["max_episdode_steps"] = 3
    with pytest.raises(ValueError, match="Unknown"):
        TaskSpec.from_dict(data)
    data = spec.to_dict()
    data["success"][0]["threshold"] = "one"
    with pytest.raises(ValueError, match="finite"):
        TaskSpec.from_dict(data)
    with pytest.raises(ValueError, match="Unknown"):
        load_task("../collect_oak_log")


def test_yaml_safe_loading(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "task.yaml"
    path.write_text("id: test\nversion: '1'\ninstruction: Test\nmax_episode_steps: 2\n")
    assert TaskSpec.from_yaml(path).max_episode_steps == 2
    path.write_text("!!python/object/apply:os.system ['echo unsafe']")
    import yaml

    with pytest.raises(yaml.YAMLError):
        TaskSpec.from_yaml(path)
