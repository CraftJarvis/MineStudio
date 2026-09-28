import numpy as np
import pytest

from minestudio.actions import noop_action
from minestudio.core import BackendError
from minestudio.rollout import MemoryRecorder, RolloutConfig, RolloutRunner


class CounterEnv:
    def __init__(self, *, fail=False, length=3):
        self.closed = False
        self.fail = fail
        self.length = length
        self.steps = 0
        self.image = np.zeros((2, 3, 3), np.uint8)

    def reset(self, *, seed=None, options=None):
        self.seed = seed
        self.steps = 0
        return {"image": self.image}, {}

    def step(self, action):
        if self.fail and self.steps == 1:
            raise BackendError("connection lost")
        self.steps += 1
        self.image.fill(self.steps)
        return {"image": self.image}, 2.0, self.steps == self.length, False, {}

    def close(self):
        self.closed = True


class CounterPolicy:
    def __init__(self):
        self.states = []
        self.closed = False

    def initial_state(self):
        return 0

    def act(self, observation, state=None, *, deterministic=False):
        self.states.append(state)
        # Mutating policy input cannot corrupt trajectory history.
        observation["image"].fill(200)
        return noop_action(), state + 1

    def close(self):
        self.closed = True


def test_global_budget_state_isolation_and_final_observation():
    envs, policies = [], []

    def env_factory():
        envs.append(CounterEnv())
        return envs[-1]

    def policy_factory():
        policies.append(CounterPolicy())
        return policies[-1]

    recorder = MemoryRecorder()
    with RolloutRunner(
        env_factory,
        policy_factory,
        RolloutConfig(num_envs=2, num_episodes=5, max_episode_steps=2),
        recorders=[recorder],
    ) as runner:
        result = runner.collect()
    assert (result.completed, result.failed, result.cancelled) == (5, 0, 0)
    assert len(envs) == len(policies) == 5
    assert all(env.closed for env in envs)
    assert all(policy.closed and policy.states == [0, 1] for policy in policies)
    for episode in recorder.episodes.values():
        assert len(episode.observations) == 3
        assert [int(obs["image"].max()) for obs in episode.observations] == [0, 1, 2]
        assert episode.transitions[-1].truncated
        assert not episode.transitions[-1].terminated
        assert episode.transitions[-1].info["truncation_reason"] == "rollout_budget"


def test_retry_same_case_seed_new_attempt_and_no_fake_transition():
    envs = []

    def factory():
        envs.append(CounterEnv(fail=not envs))
        return envs[-1]

    recorder = MemoryRecorder()
    with RolloutRunner(
        factory, CounterPolicy, RolloutConfig(num_episodes=2, max_retries=1), recorders=[recorder]
    ) as runner:
        result = runner.collect()
    assert (result.completed, result.failed) == (2, 0)
    failed = result.attempts[0]
    retry = result.attempts[-1]
    assert failed.status == "failed" and failed.num_steps == 1
    assert failed.context.case_id == retry.context.case_id
    assert failed.context.seed == retry.context.seed
    assert failed.context.attempt_id != retry.context.attempt_id
    partial = recorder.transitions[failed.context.attempt_id]
    assert len(partial) == 1
    assert not partial[0].terminated and not partial[0].truncated
    assert all(env.closed for env in envs)


def test_exhausted_failures_do_not_block_other_cases():
    with RolloutRunner(
        lambda: CounterEnv(fail=True), CounterPolicy, RolloutConfig(num_episodes=3, max_retries=1)
    ) as runner:
        result = runner.collect()
    assert (result.completed, result.failed, result.cancelled) == (0, 3, 0)
    assert len(result.attempts) == 6


def test_seed_independent_of_slot_count():
    def run(slots):
        with RolloutRunner(
            CounterEnv, CounterPolicy, RolloutConfig(num_envs=slots, num_episodes=4, seed=39)
        ) as runner:
            return {r.context.case_id: r.context.seed for r in runner.collect().attempts}

    assert run(1) == run(3)


def test_programming_error_propagates_and_closes_resources():
    env = CounterEnv()

    class BrokenPolicy(CounterPolicy):
        def act(self, *args, **kwargs):
            raise ValueError("bad model shape")

    policy = BrokenPolicy()
    with (
        RolloutRunner(lambda: env, lambda: policy) as runner,
        pytest.raises(ValueError, match="bad model shape"),
    ):
        runner.collect()
    assert env.closed and policy.closed


def test_cancel_counts_unstarted_cases():
    class CancellingRecorder(MemoryRecorder):
        def on_transition(self, context, transition):
            super().on_transition(context, transition)
            runner.cancel()

    recorder = CancellingRecorder()
    with RolloutRunner(
        CounterEnv, CounterPolicy, RolloutConfig(num_envs=2, num_episodes=5), recorders=[recorder]
    ) as runner:
        result = runner.collect()
    assert (result.completed, result.failed, result.cancelled) == (0, 0, 5)
    assert len(result.attempts) == 2
    assert all(r.status == "cancelled" for r in result.attempts)


@pytest.mark.parametrize("event", ["on_reset", "on_transition", "on_episode_end"])
def test_recorder_backend_error_is_not_retried(event):
    envs = []

    def factory():
        envs.append(CounterEnv(length=1))
        return envs[-1]

    class BrokenRecorder(MemoryRecorder):
        pass

    def fail(*args):
        raise BackendError("recorder storage failed")

    recorder = BrokenRecorder()
    setattr(recorder, event, fail)
    with (
        RolloutRunner(
            factory, CounterPolicy, RolloutConfig(max_retries=2), recorders=[recorder]
        ) as runner,
        pytest.raises(BackendError, match="recorder storage"),
    ):
        runner.collect()
    assert len(envs) == 1
    assert envs[0].closed
