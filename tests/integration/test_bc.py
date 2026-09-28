# ruff: noqa: E402
"""Numerical acceptance for differentiable likelihood and optimizer recovery."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("gym3")
pytest.importorskip("safetensors")

from minestudio.actions import noop_action
from minestudio.core import ImageSize, TrajectoryWindow
from minestudio.policies.batches import ActionEvaluation, PolicyBatch
from minestudio.policies.vpt import VPTConfig, VPTPolicy
from minestudio.training import BCConfig, train_bc
from minestudio.training.bc import bc_loss

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


def window(index=0, length=8):
    rng = np.random.default_rng(index)
    ones, zeros = np.ones(length, dtype=bool), np.zeros(length, dtype=bool)
    actions = []
    for t in range(length):
        action = noop_action()
        action["buttons"]["attack"] = t % 2
        action["camera"][:] = [t % 3, 1]
        actions.append(action)
    return TrajectoryWindow(
        str(index),
        0,
        {"image": rng.integers(0, 256, (length + 1, 19, 21, 3), dtype=np.uint8)},
        tuple(actions),
        np.zeros(length, np.float32),
        zeros,
        zeros,
        ones,
        {"image": np.ones(length + 1, bool)},
        ones,
        zeros,
        zeros,
        zeros,
    )


def slice_batch(batch, lo, hi):
    return PolicyBatch(
        {k: v[:, lo:hi] for k, v in batch.inputs.items()},
        {k: v[:, lo:hi] for k, v in batch.actions.items()},
        batch.valid_mask[:, lo:hi],
        batch.first_mask[:, lo:hi],
    )


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_sequence_step_state_parity_and_gradients(device):
    policy = tiny_policy(device)
    batch = policy.prepare_batch([window(), window(1)])
    # Cross the cache limit and reset only the second stream inside a sequence.
    batch.first_mask[1, 4] = True
    with torch.no_grad():
        whole = policy.evaluate_actions(batch)
        state, parts = None, []
        for t in range(8):
            result = policy.evaluate_actions(slice_batch(batch, t, t + 1), state)
            parts.append(result.log_prob)
            state = result.next_state
        torch.testing.assert_close(whole.log_prob, torch.cat(parts, dim=1), atol=2e-5, rtol=2e-5)
        for a, b in zip(whole.next_state, state, strict=True):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        fresh = policy.evaluate_actions(slice_batch(batch, 4, 8))
        torch.testing.assert_close(whole.log_prob[1, 4:], fresh.log_prob[1], atol=2e-5, rtol=2e-5)
    policy.train()
    (-policy.evaluate_actions(batch).log_prob.mean()).backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.net.parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in policy.pi_head.parameters()
    )
    assert all(p.grad is None for p in policy.value_head.parameters())


def test_loss_mask_normalization_and_neutral_camera(tmp_path):
    heads = {
        "buttons": torch.tensor([[-2.0, -99.0, -6.0]], requires_grad=True),
        "camera": torch.tensor([[-3.0, -99.0, -5.0]], requires_grad=True),
    }
    batch = PolicyBatch(
        {},
        {"camera": torch.tensor([[[60], [60], [1]]])},
        torch.tensor([[True, False, True]]),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    result = ActionEvaluation(sum(heads.values()), heads, torch.zeros(1, 3), None)
    config = BCConfig(tmp_path)
    assert bc_loss(result, batch, config).item() == 8
    loss = bc_loss(result, batch, replace(config, mask_neutral_camera=True))
    assert loss.item() == 6.5
    loss.backward()
    assert heads["buttons"].grad.tolist() == [[-0.5, 0.0, -0.5]]
    assert heads["camera"].grad.tolist() == [[0.0, 0.0, -0.5]]
    with pytest.raises(ValueError, match="no supervised"):
        bc_loss(result, replace(batch, valid_mask=torch.zeros(1, 3, dtype=torch.bool)), config)


def test_processor_missing_labels_and_shared_inference_resize():
    policy = tiny_policy()
    source = window(length=4)
    mask = np.array([True, False, True, True])
    missing = replace(source, action_mask=mask)
    batch = policy.prepare_batch([missing])
    assert batch.valid_mask.tolist() == [[True, False, True, True]]
    assert not batch.first_mask.any()  # A window start is not an episode start.
    with torch.no_grad():
        sampled, _ = policy.act_with_stats({"image": source.observations["image"][0]})
        batch.actions["buttons"][0, 0, 0] = sampled.policy_action["buttons"]
        batch.actions["camera"][0, 0, 0] = sampled.policy_action["camera"]
        result = policy.evaluate_actions(slice_batch(batch, 0, 1))
        assert float(result.log_prob[0, 0]) == pytest.approx(sampled.log_prob, abs=2e-6)


class InMemoryWindows:
    fingerprint = "fixed-test-windows-v1"

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return window(index, length=3)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_optimizer_resume_and_export(tmp_path, device):
    dataset = InMemoryWindows()
    original = tiny_policy(device)
    before = copy.deepcopy(original.state_dict())
    full = copy.deepcopy(original)
    config = BCConfig(
        tmp_path / "full",
        max_steps=4,
        batch_size=2,
        learning_rate=1e-3,
        validation_interval=1,
        checkpoint_interval=2,
    )
    uninterrupted = train_bc(full, dataset, config=config, validation_dataset=dataset)
    part = train_bc(
        original,
        dataset,
        config=replace(config, output_dir=tmp_path / "part", max_steps=2),
        validation_dataset=dataset,
    )
    resumed = tiny_policy(device)
    result = train_bc(
        resumed,
        dataset,
        config=replace(config, output_dir=tmp_path / "resume"),
        validation_dataset=dataset,
        resume_from=part.checkpoint,
    )
    reloaded = VPTPolicy.from_pretrained(result.export, device=device)
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(value, reloaded.state_dict()[name], rtol=0, atol=0)
    assert any(
        not torch.equal(before[n], p) for n, p in full.named_parameters() if n.startswith("net.")
    )
    assert all(
        torch.equal(before[n], p) for n, p in full.named_parameters() if n.startswith("value_head.")
    )
    full_rows = [json.loads(line) for line in uninterrupted.metrics.read_text().splitlines()]
    resumed_rows = [json.loads(line) for line in result.metrics.read_text().splitlines()]
    assert [(r["indices"], r["loss"]) for r in full_rows if "indices" in r][2:] == [
        (r["indices"], r["loss"]) for r in resumed_rows if "indices" in r
    ]
    with pytest.raises(ValueError, match="differs"):
        train_bc(
            tiny_policy(device),
            dataset,
            config=replace(config, output_dir=tmp_path / "bad", camera_weight=2),
            validation_dataset=dataset,
            resume_from=part.checkpoint,
        )
