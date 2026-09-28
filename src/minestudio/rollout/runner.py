"""Synchronous interleaved rollout with per-attempt ownership and bounded retries."""

import logging
import math
import sys
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

import numpy as np

from minestudio.actions import copy_action
from minestudio.core import BackendError, Observation, Transition
from minestudio.core.types import snapshot_observation
from minestudio.envs.protocols import Environment
from minestudio.policies.protocols import Policy
from minestudio.rollout.config import RolloutConfig
from minestudio.rollout.records import (
    EpisodeContext,
    EpisodeResult,
    Recorder,
    RolloutCase,
    RolloutResult,
)

logger = logging.getLogger(__name__)
StateT = TypeVar("StateT")


@runtime_checkable
class _Closable(Protocol):
    def close(self) -> None: ...


def _close(resource: _Closable) -> None:
    try:
        resource.close()
    except Exception:
        logger.exception("Rollout resource cleanup failed")


@dataclass
class _Attempt(Generic[StateT]):
    context: EpisodeContext
    retries: int
    resources: ExitStack
    env: Environment
    policy: Policy[StateT]
    observation: Observation
    state: StateT
    steps: int = 0
    total_reward: float = 0.0
    ended: bool = False


class RolloutRunner(Generic[StateT]):
    """Collect a global case quota using at most num_envs interleaved local slots.

    Factories must create fresh owned resources. BackendError retries a case with
    the same seed and a new attempt id. Programming/configuration errors propagate.
    Policy RNGs are independent of environment seeds; seed them in policy_factory
    if stochastic action reproducibility is required.
    """

    def __init__(
        self,
        env_factory: Callable[[], Environment],
        policy_factory: Callable[[], Policy[StateT]],
        config: RolloutConfig | None = None,
        *,
        recorders: Sequence[Recorder] = (),
        cases: Sequence[RolloutCase] | None = None,
    ) -> None:
        self.config = config or RolloutConfig()
        self._env_factory = env_factory
        self._policy_factory = policy_factory
        self._recorders = tuple(recorders)
        self._cases = (
            tuple(cases)
            if cases is not None
            else tuple(
                RolloutCase(
                    case, int(np.random.SeedSequence([self.config.seed, case]).generate_state(1)[0])
                )
                for case in range(self.config.num_episodes)
            )
        )
        if len(self._cases) != self.config.num_episodes or len(
            {c.case_id for c in self._cases}
        ) != len(self._cases):
            raise ValueError("cases must have unique ids and match num_episodes")
        self._closed = False
        self._running = False
        self._cancelled = False

    def cancel(self) -> None:
        """Stop at the next step boundary; unfinished cases count as cancelled."""
        self._cancelled = True

    def collect(self, *, run_id: str | None = None) -> RolloutResult:
        if self._closed or self._running:
            raise RuntimeError("Runner is closed or already collecting")
        run_id = uuid4().hex if run_id is None else run_id
        if not run_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in run_id
        ):
            raise ValueError("run_id must contain only letters, digits, '_' and '-'")
        self._running = True
        self._cancelled = False
        pending = deque((case, 0) for case in range(self.config.num_episodes))
        active: list[_Attempt[StateT]] = []
        results: list[EpisodeResult] = []
        completed = failed = 0

        def finish(result: EpisodeResult) -> None:
            results.append(result)
            for recorder in self._recorders:
                recorder.on_episode_end(result)

        try:
            while (pending or active) and not self._cancelled:
                while pending and len(active) < self.config.num_envs and not self._cancelled:
                    case, retries = pending.popleft()
                    case_spec = self._cases[case]
                    seed = case_spec.seed
                    context = EpisodeContext(
                        run_id, case_spec.case_id, uuid4().hex, seed, deepcopy(case_spec.metadata)
                    )
                    resources = ExitStack()
                    for recorder in self._recorders:
                        recorder.on_episode_start(context)
                    try:
                        env = (case_spec.env_factory or self._env_factory)()
                        resources.callback(_close, env)
                        policy = self._policy_factory()
                        if isinstance(policy, _Closable):
                            resources.callback(_close, policy)
                        observation, info = env.reset(seed=seed)
                        observation = snapshot_observation(observation)
                        state = policy.initial_state()
                    except BackendError as error:
                        resources.close()
                        finish(
                            EpisodeResult(
                                context,
                                "failed",
                                0,
                                0.0,
                                error=str(error),
                                end_reason="backend_error",
                                error_type=type(error).__name__,
                            )
                        )
                        if retries < self.config.max_retries:
                            pending.append((case, retries + 1))
                        else:
                            failed += 1
                    except BaseException as error:
                        resources.close()
                        for recorder in self._recorders:
                            try:
                                recorder.on_episode_end(
                                    EpisodeResult(
                                        context,
                                        "failed",
                                        0,
                                        0.0,
                                        error=str(error),
                                        end_reason="exception",
                                        error_type=type(error).__name__,
                                    )
                                )
                            except Exception:
                                logger.exception("Recorder failed while finalizing a failed reset")
                        raise
                    else:
                        active.append(
                            _Attempt(context, retries, resources, env, policy, observation, state)
                        )
                        for recorder in self._recorders:
                            recorder.on_reset(context, deepcopy(observation), deepcopy(info))
                for attempt in active[:]:
                    if self._cancelled:
                        break
                    action, next_state = attempt.policy.act(
                        snapshot_observation(attempt.observation),
                        attempt.state,
                        deterministic=self.config.deterministic,
                    )
                    action = copy_action(action)
                    try:
                        observation, reward, terminated, truncated, info = attempt.env.step(
                            copy_action(action)
                        )
                    except BackendError as error:
                        attempt.resources.close()
                        active.remove(attempt)
                        finish(
                            EpisodeResult(
                                attempt.context,
                                "failed",
                                attempt.steps,
                                attempt.total_reward,
                                error=str(error),
                                end_reason="backend_error",
                                error_type=type(error).__name__,
                            )
                        )
                        if attempt.retries < self.config.max_retries:
                            pending.append(
                                (
                                    next(
                                        i
                                        for i, item in enumerate(self._cases)
                                        if item.case_id == attempt.context.case_id
                                    ),
                                    attempt.retries + 1,
                                )
                            )
                        else:
                            failed += 1
                        continue
                    if (
                        not math.isfinite(reward)
                        or type(terminated) is not bool
                        or type(truncated) is not bool
                    ):
                        raise ValueError("Environment returned an invalid reward/boundary")
                    observation = snapshot_observation(observation)
                    info = deepcopy(info)
                    attempt.steps += 1
                    attempt.total_reward += reward
                    if attempt.steps >= self.config.max_episode_steps and not (
                        terminated or truncated
                    ):
                        truncated = True
                        info["truncation_reason"] = "rollout_budget"
                        info["end_reason"] = "rollout_budget"
                    transition = Transition(
                        attempt.observation,
                        action,
                        reward,
                        observation,
                        terminated,
                        truncated,
                        info,
                    )
                    attempt.observation, attempt.state = observation, next_state
                    for recorder in self._recorders:
                        recorder.on_transition(attempt.context, deepcopy(transition))
                    if terminated or truncated:
                        success = info.get("success")
                        if success is not None and type(success) is not bool:
                            raise ValueError("success must be a boolean or None")
                        reason = info.get("end_reason") or (
                            "backend_terminal" if terminated else "backend_truncation"
                        )
                        if not isinstance(reason, str):
                            raise ValueError("end_reason must be a string")
                        attempt.ended = True
                        attempt.resources.close()
                        active.remove(attempt)
                        finish(
                            EpisodeResult(
                                attempt.context,
                                "completed",
                                attempt.steps,
                                attempt.total_reward,
                                terminated,
                                truncated,
                                success=success,
                                end_reason=reason,
                            )
                        )
                        completed += 1
            if self._cancelled:
                for attempt in active:
                    attempt.ended = True
                    attempt.resources.close()
                    finish(
                        EpisodeResult(
                            attempt.context,
                            "cancelled",
                            attempt.steps,
                            attempt.total_reward,
                            end_reason="cancelled",
                        )
                    )
            return RolloutResult(
                run_id,
                tuple(results),
                completed,
                failed,
                self.config.num_episodes - completed - failed,
            )
        finally:
            error = sys.exc_info()[1]
            for attempt in active:
                attempt.resources.close()
                if error is not None and not attempt.ended:
                    attempt.ended = True
                    result = EpisodeResult(
                        attempt.context,
                        "failed",
                        attempt.steps,
                        attempt.total_reward,
                        error=str(error),
                        end_reason="exception",
                        error_type=type(error).__name__,
                    )
                    for recorder in self._recorders:
                        try:
                            recorder.on_episode_end(result)
                        except Exception:
                            logger.exception("Recorder failed while finalizing an aborted attempt")
            self._running = False

    def close(self) -> None:
        self.cancel()
        self._closed = True

    def __enter__(self) -> "RolloutRunner[StateT]":
        if self._closed:
            raise RuntimeError("Runner is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
