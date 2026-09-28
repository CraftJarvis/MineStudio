# ruff: noqa: E402
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("gym3")
pytest.importorskip("safetensors")
from minestudio.actions import validate_action
from minestudio.core import ImageSize
from minestudio.policies.vpt import VPTConfig, VPTPolicy

pytestmark = pytest.mark.torch


def tiny_policy():
    torch.set_num_threads(1)
    return VPTPolicy(
        VPTConfig(
            image_size=ImageSize(16, 16),
            hidden_size=16,
            attention_heads=2,
            attention_memory_size=4,
            num_layers=1,
        )
    )


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_vpt_inference_state_and_verified_export(tmp_path, device):
    policy = tiny_policy().to(device)
    obs = {"image": np.zeros((20, 24, 3), np.uint8)}
    initial = policy.initial_state()
    result, state = policy.act_with_stats(obs, initial, deterministic=True)
    validate_action(result.action)
    assert initial.first and not state.first
    assert all(t.device == torch.device(device) for t in state.recurrent)
    assert all(p.device == torch.device(device) for p in policy.parameters())
    assert np.isfinite(result.value)
    assert result.log_prob == 0
    other, _ = policy.act_with_stats(obs, policy.initial_state(), deterministic=True)
    assert result.policy_action == other.policy_action
    destination = tmp_path / "export"
    policy.save_pretrained(destination)
    loaded = VPTPolicy.from_pretrained(destination, device=device)
    loaded_result, _ = loaded.act_with_stats(obs, deterministic=True)
    assert loaded_result.policy_action == result.policy_action
    assert loaded_result.value == pytest.approx(result.value)
    with (destination / "model.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        VPTPolicy.from_pretrained(destination)


def test_legacy_import_rejects_missing_weights():
    policy = tiny_policy()
    weights = dict(policy.state_dict())
    weights.pop(next(iter(weights)))
    with pytest.raises(RuntimeError, match="Missing key"):
        VPTPolicy.from_legacy_state_dict(weights, config=policy.config)


def test_convert_hub_config_and_tensor_weights(tmp_path):
    from safetensors.torch import save_file

    from minestudio.policies.vpt.conversion import convert_legacy_export

    policy = tiny_policy()
    source = {
        "action_space": None,
        "policy_kwargs": policy.config.network_kwargs(),
        "temperature": 1,
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(source))
    weights = tmp_path / "source.safetensors"
    save_file(policy.state_dict(), str(weights))
    destination = convert_legacy_export(config, weights, tmp_path / "export")
    loaded = VPTPolicy.from_pretrained(destination)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["provenance"]["source_config"] == json.loads(config.read_text())
    source["action_space"] = {"buttons": 1}
    config.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="action_space"):
        convert_legacy_export(config, weights, tmp_path / "unsupported")


def test_native_manifest_binds_processor_and_codec(tmp_path):
    policy = tiny_policy()
    destination = tmp_path / "export"
    policy.save_pretrained(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    manifest["codec"] = "unknown"
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="codec"):
        VPTPolicy.from_pretrained(destination)


def test_network_outputs_match_original_v1_with_preserved_cache_and_norm():
    """Run original v1 class definitions without importing its unrelated training dependencies."""
    import ast
    from copy import deepcopy
    from importlib.resources import files
    from typing import Optional

    from minestudio.policies.vpt._network import (
        FanInInitReLULayer,
        ImpalaCNN,
        ResidualRecurrentBlocks,
    )

    source = (files("minestudio") / "models/vpt/body.py").read_text()
    original = ast.parse(source)
    names = {"ImgPreprocessing", "ImgObsProcess", "MinecraftPolicy"}
    classes = ast.Module(
        body=[
            node for node in original.body if isinstance(node, ast.ClassDef) and node.name in names
        ],
        type_ignores=[],
    )
    namespace = {
        "nn": torch.nn,
        "th": torch,
        "F": torch.nn.functional,
        "np": np,
        "deepcopy": deepcopy,
        "Optional": Optional,
        "Dict": dict,
        "ImpalaCNN": ImpalaCNN,
        "FanInInitReLULayer": FanInInitReLULayer,
        "ResidualRecurrentBlocks": ResidualRecurrentBlocks,
    }
    exec(compile(classes, "v1/body.py", "exec"), namespace)
    legacy_kwargs = {
        "hidsize": 16,
        "img_shape": [16, 16, 3],
        "attention_heads": 2,
        "attention_memory_size": 6,
        "timesteps": 2,
        "n_recurrence_layers": 1,
        "init_norm_kwargs": {"group_norm_groups": 1, "batch_norm": False},
        "impala_kwargs": {"post_pool_groups": 1},
        "use_pre_lstm_ln": False,
    }
    config = VPTConfig.from_legacy_kwargs(legacy_kwargs)
    policy = VPTPolicy(config).eval()
    original_network = namespace["MinecraftPolicy"](**legacy_kwargs).eval()
    original_network.load_state_dict(policy.net.state_dict(), strict=True)
    old_state, new_state = original_network.initial_state(1), policy.initial_state()
    rng = np.random.default_rng(13)
    with torch.inference_mode():
        for step in range(5):
            observation = {"image": rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)}
            image = torch.from_numpy(observation["image"])[None, None]
            first = torch.tensor([[step == 0]])
            (old_pi, old_value), old_state = original_network(
                {"image": image}, old_state, {"first": first}
            )
            (new_pi, new_value), _ = policy.net(
                {"image": image}, deepcopy(new_state.recurrent), {"first": first}
            )
            torch.testing.assert_close(new_pi, old_pi, rtol=0, atol=0)
            torch.testing.assert_close(new_value, old_value, rtol=0, atol=0)
            old_logits = policy.pi_head(old_pi)
            old_action = policy.pi_head.sample(old_logits, deterministic=True)
            actual, new_state = policy.act_with_stats(observation, new_state, deterministic=True)
            assert actual.policy_action == {
                key: int(value.item()) for key, value in old_action.items()
            }
            assert actual.value == pytest.approx(
                float(policy.value_head.denormalize(policy.value_head(old_value)).item())
            )
            for old, new in zip(old_state, new_state.recurrent, strict=True):
                torch.testing.assert_close(new, old, rtol=0, atol=0)
