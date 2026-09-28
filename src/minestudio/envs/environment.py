"""Gymnasium environment with explicit resource ownership and episode boundaries."""

import logging
import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal, cast

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as error:
    from minestudio.core import MissingDependencyError

    raise MissingDependencyError("Install minestudio[envs] to use MinecraftEnv") from error
from minestudio.actions import BUTTONS, copy_action
from minestudio.core import EpisodeStateError, Info, MinecraftAction, Observation, StepResult
from minestudio.core.types import snapshot_observation
from minestudio.envs.config import EnvConfig
from minestudio.envs.protocols import EnvBackend
from minestudio.tasks import TaskRuntime, TaskSpec

logger = logging.getLogger(__name__)


def _minerl_factory(config: EnvConfig) -> EnvBackend:
    from minestudio.envs.backends.minerl import MineRLBackend

    return MineRLBackend(config)


class MinecraftEnv(gym.Env[Observation, MinecraftAction]):
    """Start the backend on first reset. Never download assets or auto-reset.

    backend_factory is an extension/test seam. Its fresh backend is owned by this
    environment and must return observations matching config.image_size.
    """

    def __init__(
        self,
        config: EnvConfig | None = None,
        *,
        render_mode: str | None = None,
        task: TaskSpec | None = None,
        backend_factory: Callable[[EnvConfig], EnvBackend] = _minerl_factory,
    ) -> None:
        super().__init__()
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.task = task
        self._task_runtime = TaskRuntime(task) if task is not None else None
        self._factory = backend_factory
        self._backend: EnvBackend | None = None
        self._closed = False
        self._active = False
        self._steps = 0
        self._observation: Observation | None = None
        self.observation_space = cast(
            gym.Space[Observation],
            spaces.Dict(
                {
                    "image": spaces.Box(
                        0,
                        255,
                        (self.config.image_size.height, self.config.image_size.width, 3),
                        np.uint8,
                    )
                }
            ),
        )
        self.action_space = cast(
            gym.Space[MinecraftAction],
            spaces.Dict(
                {
                    "buttons": spaces.Dict({name: spaces.Discrete(2) for name in BUTTONS}),
                    "camera": spaces.Box(-np.inf, np.inf, (2,), np.float32),
                }
            ),
        )

    def _snapshot(self, observation: Observation) -> Observation:
        result = snapshot_observation(observation)
        if not self.observation_space.contains(result):
            raise ValueError("Backend observation does not match config.image_size")
        self._observation = snapshot_observation(result)
        return result

    def reset(
        self, *, seed: int | None = None, options: Info | None = None
    ) -> tuple[Observation, Info]:
        if self._closed:
            raise EpisodeStateError("Cannot reset a closed environment")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        super().reset(seed=seed)
        effective_seed = (
            seed if seed is not None else int.from_bytes(self.np_random.bytes(4), "little")
        )
        self.action_space.seed(effective_seed)
        self._active = False
        self._observation = None
        try:
            if self._backend is None:
                self._backend = self._factory(self.config)
            observation, info = self._backend.reset(seed=effective_seed, options=deepcopy(options))
            if self._task_runtime is not None:
                observation, info = self._task_runtime.initialize(self._backend, observation, info)
            result = self._snapshot(observation)
        except BaseException:
            self._discard_backend()
            raise
        self._steps = 0
        self._active = True
        return result, {**deepcopy(info), "seed": effective_seed}

    def step(self, action: MinecraftAction) -> StepResult:
        if not self._active or self._backend is None:
            raise EpisodeStateError(
                "Call reset before stepping, including after an episode boundary"
            )
        owned_action = copy_action(action)
        try:
            observation, reward, terminated, truncated, info = self._backend.step(owned_action)
            if not math.isfinite(reward):
                raise ValueError("Backend returned a non-finite reward")
            if self._task_runtime is not None:
                task_step = self._task_runtime.step(info, reward, terminated, truncated)
                reward, terminated, truncated, info = (
                    task_step.reward,
                    task_step.terminated,
                    task_step.truncated,
                    task_step.info,
                )
            result = self._snapshot(observation)
        except BaseException:
            self._active = False
            self._discard_backend()
            raise
        self._steps += 1
        info = deepcopy(info)
        if (
            self.config.max_episode_steps is not None
            and self._steps >= self.config.max_episode_steps
            and not (terminated or truncated)
        ):
            truncated = True
            info["truncation_reason"] = "time_limit"
            info["end_reason"] = "time_limit"
        self._active = not (terminated or truncated)
        return result, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        """Return a copy of the last policy-resolution image when rgb_array is enabled."""
        if self.render_mode != "rgb_array":
            return None
        if self._observation is None:
            raise EpisodeStateError("Call reset before render")
        return self._observation["image"].copy()

    def _discard_backend(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            try:
                backend.close()
            except Exception:
                logger.exception("Backend cleanup failed")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._active = False
            self._observation = None
            self._discard_backend()

    def __enter__(self) -> "MinecraftEnv":
        if self._closed:
            raise EpisodeStateError("Environment is closed")
        return self

    def __exit__(self, *args: object) -> Literal[False]:
        self.close()
        return False
