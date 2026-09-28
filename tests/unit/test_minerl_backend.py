"""Exercise adapter translation without starting MineRL or importing its runtime."""

import numpy as np
import pytest

from minestudio.actions import noop_action
from minestudio.core import BackendError, ImageSize
from minestudio.envs import EnvConfig
from minestudio.envs.backends.minerl import MineRLBackend


class LegacyEnv:
    def __init__(self):
        self.error = False
        self.done = False

    def noop_action(self):
        return {"chat": ""}

    def step(self, action):
        self.received = action
        return (
            {
                "pov": np.zeros((4, 6, 3), np.uint8),
                "inventory": {0: {"type": "minecraft:oak_log", "quantity": 2}},
            },
            1,
            self.done,
            {"error": "lost socket"} if self.error else {},
        )

    def execute_cmd(self, command):
        return self.step({"chat": command})

    def close(self):
        pass


def backend():
    instance = object.__new__(MineRLBackend)
    instance.config = EnvConfig(image_size=ImageSize(4, 6))
    instance._env = LegacyEnv()
    instance._resize = lambda image, size: image
    instance._engine_identity = {"jar_sha256": "test-only"}
    return instance


def test_adapter_native_controls_metrics_and_provenance():
    instance = backend()
    action = noop_action()
    action["buttons"]["drop"] = 1
    _, reward, terminated, truncated, info = instance.step(action)
    assert instance._env.received["drop"] == 1
    assert instance._env.received["chat"] == ""
    assert info["metrics"]["inventory"] == {"oak_log": 2}
    assert "pov" not in info["backend_observation"]
    assert info["engine"]["jar_sha256"] == "test-only"
    assert (reward, terminated, truncated) == (1, False, False)
    instance._env.received["camera"][0] = 9
    assert action["camera"][0] == 0


def test_legacy_fake_terminal_error_is_never_returned():
    instance = backend()
    instance._env.error = True
    with pytest.raises(BackendError, match="lost socket"):
        instance.step(noop_action())
    instance._env.error = False
    instance._env.done = True
    with pytest.raises(BackendError, match="initialization"):
        instance.execute_command("/time set day")
