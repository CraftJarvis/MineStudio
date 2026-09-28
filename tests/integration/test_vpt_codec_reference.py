import numpy as np
import pytest

pytest.importorskip("attr")
from minestudio.actions import VPTActionCodec, noop_action
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.utils.vpt_lib.actions import ActionTransformer, Buttons


def test_codec_matches_legacy_on_conflicts_clipping_and_all_decode_indices():
    codec = VPTActionCodec()
    mapping = CameraHierarchicalMapping(n_camera_bins=11)
    transformer = ActionTransformer(
        camera_binsize=2, camera_maxval=10, camera_mu=10, camera_quantization_scheme="mu_law"
    )
    rng = np.random.default_rng(32)
    for _ in range(256):
        action = noop_action()
        action["buttons"].update({name: int(rng.integers(0, 2)) for name in Buttons.ALL})
        action["camera"] = rng.uniform(-30, 30, 2).astype(np.float32)
        flat = {**action["buttons"], "camera": action["camera"]}
        factored = transformer.env2policy(flat)
        encoded = mapping.from_factored({key: value[None] for key, value in factored.items()})
        assert codec.encode(action) == {key: int(value.item()) for key, value in encoded.items()}
    for index in range(8641):
        camera = index % 121
        legacy = mapping.to_factored(
            {"buttons": np.array([[index]]), "camera": np.array([[camera]])}
        )
        native = transformer.policy2env(legacy)
        decoded = codec.decode({"buttons": index, "camera": camera})
        assert decoded["buttons"] == {name: int(native[name].item()) for name in Buttons.ALL}
        np.testing.assert_allclose(decoded["camera"], native["camera"][0], atol=2e-6)
