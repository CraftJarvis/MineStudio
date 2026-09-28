import numpy as np
import pytest

from minestudio.actions import noop_action
from minestudio.core import BackendError, EpisodeStateError, ImageSize
from minestudio.envs import EnvConfig, MinecraftEnv


class BufferBackend:
    def __init__(self, config):
        self.image = np.zeros((config.image_size.height, config.image_size.width, 3), np.uint8)
        self.closed = 0
        self.steps = 0
        self.info = {"nested": {"step": 0}}
        self.fail = False

    def reset(self, *, seed=None, options=None):
        self.seed = seed
        self.steps = 0
        self.image.fill(0)
        return {"image": self.image}, self.info

    def step(self, action):
        if self.fail:
            raise BackendError("disconnected")
        self.steps += 1
        self.image.fill(self.steps)
        self.info["nested"]["step"] = self.steps
        action["camera"].fill(42)
        return {"image": self.image}, 1.0, self.steps == 2, False, self.info

    def close(self):
        self.closed += 1


def test_lazy_lifecycle_snapshot_and_no_autoreset():
    created = []

    def factory(config):
        backend = BufferBackend(config)
        created.append(backend)
        return backend

    env = MinecraftEnv(
        EnvConfig(image_size=ImageSize(8, 12)), backend_factory=factory, render_mode="rgb_array"
    )
    assert not created
    with pytest.raises(EpisodeStateError):
        env.step(noop_action())
    observation, info = env.reset(seed=7)
    action = noop_action()
    first = env.step(action)
    final = env.step(action)
    assert final[2:4] == (True, False)
    assert created[0].seed == info["seed"] == 7
    assert observation["image"].max() == 0
    assert info["nested"]["step"] == 0
    assert first[0]["image"].max() == 1
    assert final[0]["image"].max() == 2
    assert action["camera"].max() == 0
    rendered = env.render()
    rendered.fill(100)
    assert env.render().max() == 2
    with pytest.raises(EpisodeStateError):
        env.step(action)
    env.close()
    env.close()
    assert created[0].closed == 1
    with pytest.raises(EpisodeStateError):
        env.reset()


def test_env_budget_is_truncation():
    config = EnvConfig(max_episode_steps=1)
    with MinecraftEnv(config, backend_factory=BufferBackend) as env:
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(noop_action())
        assert (terminated, truncated) == (False, True)
        assert info["truncation_reason"] == "time_limit"


def test_failure_closes_backend_and_requires_reset():
    backend = BufferBackend(EnvConfig())
    with MinecraftEnv(backend_factory=lambda _: backend) as env:
        env.reset()
        backend.fail = True
        with pytest.raises(BackendError):
            env.step(noop_action())
        assert backend.closed == 1
        with pytest.raises(EpisodeStateError):
            env.step(noop_action())
    assert backend.closed == 1


def test_failed_reset_releases_backend():
    class FailingBackend(BufferBackend):
        def reset(self, **kwargs):
            raise BackendError("startup failed")

    backend = FailingBackend(EnvConfig())
    with MinecraftEnv(backend_factory=lambda _: backend) as env, pytest.raises(BackendError):
        env.reset()
    assert backend.closed == 1


def test_gymnasium_sample_can_be_stepped():
    with MinecraftEnv(backend_factory=BufferBackend) as env:
        env.reset(seed=0)
        action = env.action_space.sample()
        # Infinite camera bounds sample a normal distribution.
        assert env.action_space.contains(action)
        env.step(action)
