# ruff: noqa: E402
"""PPO numerical semantics, sampled-code preservation and round-boundary recovery."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gym3")
pytest.importorskip("safetensors")
pytest.importorskip("cv2")

from minestudio.actions import noop_action
from minestudio.core import ImageSize
from minestudio.policies.batches import ActorCriticEvaluation, PolicyBatch
from minestudio.policies.vpt import VPTConfig, VPTPolicy
from minestudio.training import PPOConfig, train_ppo
from minestudio.training.ppo import check_behavior
from minestudio.training.ppo_objective import compute_gae, ppo_loss
from minestudio.training.ppo_rollout import collect_ppo_rollout

pytestmark = pytest.mark.torch


def tiny_policy(device="cpu"):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return VPTPolicy(
        VPTConfig(
            image_size=ImageSize(16, 16),
            hidden_size=16,
            attention_heads=2,
            attention_memory_size=4,
            num_layers=1,
        )
    ).to(device)


def test_gae_terminal_truncation_and_episode_isolation():
    rewards, values, next_values = (
        torch.tensor([x])
        for x in ([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [20.0, 100.0, 40.0, 999.0])
    )
    adv, returns = compute_gae(
        rewards,
        values,
        next_values,
        torch.tensor([[False, False, False, True]]),
        torch.tensor([[False, True, False, True]]),
        gamma=0.9,
        gae_lambda=0.8,
    )
    torch.testing.assert_close(adv, torch.tensor([[60.84, 72.0, -16.92, -36.0]]))
    torch.testing.assert_close(returns, torch.tensor([[70.84, 92.0, 13.08, 4.0]]))


def test_ppo_clipping_value_scale_and_mask(tmp_path):
    log_prob = torch.tensor([[1.5, 0.5, 1.0]]).log().requires_grad_()
    value = torch.tensor([[1.0, -1.0, 99.0]], requires_grad=True)
    entropy = torch.full((1, 3), 0.5)
    evaluation = ActorCriticEvaluation(log_prob, {}, entropy, None, value)
    batch = PolicyBatch(
        {}, {}, torch.tensor([[True, True, False]]), torch.zeros(1, 3, dtype=torch.bool)
    )
    config = PPOConfig(tmp_path, "test", entropy_coefficient=0.1)
    result = ppo_loss(
        evaluation,
        batch,
        old_log_prob=torch.zeros(1, 3),
        old_value=torch.zeros(1, 3),
        advantages=torch.tensor([[2.0, -2.0, float("nan")]]),
        returns=torch.tensor([[2.0, -2.0, float("nan")]]),
        config=config,
    )
    assert float(result.policy_loss.detach()) == pytest.approx(-0.4)
    assert float(result.value_loss.detach()) == pytest.approx(1.62)
    assert float(result.loss.detach()) == pytest.approx(0.36)
    assert float(result.clip_fraction.detach()) == 1
    result.loss.backward()
    assert value.grad[0, 2] == log_prob.grad[0, 2] == 0
    assert torch.isfinite(value.grad).all()


class ShortEnvironment:
    def __init__(self, *, fail=False):
        self.closed = False
        self.fail = fail
        self.seeds = []

    def reset(self, *, seed=None):
        self.seeds.append(seed)
        self.t = 0
        self.base = (seed or 0) % 90
        return {"image": np.full((16, 16, 3), self.base, np.uint8)}, {}

    def step(self, action):
        if self.fail:
            raise RuntimeError("engine failed")
        self.t += 1
        reward = 1.0 if action["buttons"]["attack"] else -1.0
        return (
            {"image": np.full((16, 16, 3), self.base + self.t, np.uint8)},
            reward,
            self.t == 3,
            False,
            {},
        )

    def close(self):
        self.closed = True


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_sampled_codes_and_values_reconstruct_across_resets(tmp_path, device):
    policy = tiny_policy(device)
    env = ShortEnvironment()
    config = PPOConfig(tmp_path, "short-v1", rollout_steps=8, sequence_length=4)
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device != "cpu" else None
    obs = env.reset(seed=0)[0]
    policy.value(obs)
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    rollout = collect_ppo_rollout(policy, [env], config=config, policy_version=2)
    result = check_behavior(policy, rollout, config.behavior_tolerance)
    assert result["initial_ratio_error"] < 5e-4
    assert env.seeds[-3:] == [42, 43, 44]
    assert rollout.sequences[0].batch.first_mask.tolist() == [[True, False, False, True]]
    assert rollout.sequences[1].batch.first_mask.tolist() == [[False, False, True, False]]
    assert rollout.sequences[-1].truncated[0, -1]
    assert not rollout.sequences[-1].terminated[0, -1]
    assert any(s.batch.actions["camera"].ne(60).any() for s in rollout.sequences)
    broken = replace(
        rollout, sequences=(replace(rollout.sequences[0], state_version=1), *rollout.sequences[1:])
    )
    with pytest.raises(ValueError, match="versions"):
        check_behavior(policy, broken, config.behavior_tolerance)
    corrupted = replace(
        rollout,
        sequences=(
            replace(rollout.sequences[0], old_log_prob=rollout.sequences[0].old_log_prob + 1),
            *rollout.sequences[1:],
        ),
    )
    with pytest.raises(ValueError, match="reconstruction"):
        check_behavior(policy, corrupted, config.behavior_tolerance)


class ValueProbe:
    training = False

    def initial_state(self):
        return 0

    def batch_state(self, state, *, device):
        return [torch.tensor([state])]

    def value(self, observation, state):
        return float(observation["image"][0, 0, 0]) + state

    def act_with_stats(self, observation, state, *, deterministic):
        return SimpleNamespace(
            action=noop_action(),
            policy_action={"buttons": 0, "camera": 7},
            log_prob=-1.0,
            value=self.value(observation, state),
        ), state + 1

    def prepare_sequence(self, observations, codes, first):
        return PolicyBatch(
            {"image": torch.tensor(np.stack([o["image"] for o in observations]))[None]},
            {k: torch.tensor([[[a[k]] for a in codes]]) for k in ["buttons", "camera"]},
            torch.ones(1, len(first), dtype=torch.bool),
            torch.tensor([first]),
        )


class TruncatingEnvironment:
    def reset(self, *, seed=None):
        return {"image": np.full((2, 2, 3), 100, np.uint8)}, {}

    def step(self, action):
        return {"image": np.full((2, 2, 3), 5, np.uint8)}, 2.0, False, True, {}


def test_bootstrap_uses_final_observation_and_exact_codes(tmp_path):
    config = PPOConfig(
        tmp_path, "trunc-v1", rollout_steps=2, sequence_length=2, normalize_advantage=False
    )
    rollout = collect_ppo_rollout(
        ValueProbe(), [TruncatingEnvironment()], config=config, policy_version=0
    )
    seq = rollout.sequences[0]
    assert seq.next_values.tolist() == [
        [6.0, 6.0]
    ]  # final image 5, advanced cache 1; reset image is 100
    assert seq.batch.actions["camera"].tolist() == [
        [[7], [7]]
    ]  # Native no-op would re-encode as 60.
    torch.testing.assert_close(seq.returns, torch.tensor([[2 + 0.99 * 6, 2 + 0.99 * 6]]))


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b, strict=True))
    return a == b


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_round_resume_optimizer_value_updates_and_export(tmp_path, device):
    base = tiny_policy(device)
    before = copy.deepcopy(base.state_dict())
    envs = []

    def factory():
        env = ShortEnvironment()
        envs.append(env)
        return env

    config = PPOConfig(
        tmp_path / "full",
        "short-v1",
        iterations=2,
        num_envs=2,
        rollout_steps=8,
        sequence_length=4,
        minibatch_sequences=2,
        update_epochs=2,
        learning_rate=1e-3,
        target_kl=None,
        save_rollouts=True,
    )
    full = train_ppo(lambda: copy.deepcopy(base), factory, config=config)
    first = train_ppo(
        lambda: copy.deepcopy(base),
        factory,
        config=replace(config, output_dir=tmp_path / "first", iterations=1),
    )
    resumed = train_ppo(
        lambda: copy.deepcopy(base),
        factory,
        config=replace(config, output_dir=tmp_path / "resumed"),
        resume_from=first.checkpoint,
    )
    a, b = [
        torch.load(p.checkpoint, map_location="cpu", weights_only=True) for p in (full, resumed)
    ]
    assert equal_tree(a, b)
    assert resumed.environment_steps == 32 and resumed.optimizer_steps == 8
    assert all(env.closed for env in envs)
    for prefix in ("net.", "pi_head.", "value_head.linear."):
        assert any(
            not torch.equal(before[k].cpu(), v)
            for k, v in b["model"].items()
            if k.startswith(prefix)
        )
    for k, v in b["model"].items():
        if k.startswith("value_head.normalizer."):
            assert torch.equal(v, before[k].cpu())
    loaded = VPTPolicy.from_pretrained(resumed.export, device=device)
    assert all(torch.equal(v, loaded.state_dict()[k].cpu()) for k, v in b["model"].items())
    rollout = torch.load(tmp_path / "full/rollout-000000.pt", weights_only=True)
    assert rollout["policy_version"] == 0 and len(rollout["sequences"]) == 4
    assert json.loads(resumed.config_path.read_text())["identity"]["context"].startswith(
        "reset_each_round"
    )


def test_failed_environment_closes_without_update(tmp_path):
    env = ShortEnvironment(fail=True)
    with pytest.raises(RuntimeError, match="engine failed"):
        train_ppo(
            tiny_policy,
            lambda: env,
            config=PPOConfig(tmp_path / "failure", "broken-v1", rollout_steps=4, sequence_length=2),
        )
    assert env.closed
    assert not (tmp_path / "failure/checkpoint.pt").exists()


def test_kl_stop_discards_update_and_checkpoint_rejects_changed_environment(tmp_path):
    config = PPOConfig(
        tmp_path / "kl",
        "short-v1",
        iterations=1,
        rollout_steps=8,
        sequence_length=4,
        minibatch_sequences=1,
        update_epochs=3,
        learning_rate=1e-2,
        target_kl=1e-12,
    )
    result = train_ppo(tiny_policy, ShortEnvironment, config=config)
    rows = [json.loads(line) for line in result.metrics.read_text().splitlines()]
    assert any(row["phase"] == "kl_stop" for row in rows)
    assert 0 < result.optimizer_steps < 6
    with pytest.raises(ValueError, match="differs"):
        train_ppo(
            tiny_policy,
            ShortEnvironment,
            config=replace(
                config, output_dir=tmp_path / "bad", environment_id="changed", iterations=2
            ),
            resume_from=result.checkpoint,
        )
