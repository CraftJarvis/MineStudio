import numpy as np
import pytest

from minestudio.actions import BUTTONS, VPTActionCodec, copy_action, noop_action, validate_action


def test_native_actions_are_independent_and_complete():
    action = noop_action()
    assert len(BUTTONS) == 20
    clone = copy_action(action)
    clone["camera"][0] = 4
    clone["buttons"]["forward"] = 1
    assert action["camera"][0] == 0
    assert action["buttons"]["forward"] == 0


@pytest.mark.parametrize(
    "camera", [np.zeros(3, np.float32), np.zeros(2, np.float64), np.array([np.nan, 0], np.float32)]
)
def test_invalid_camera_rejected(camera):
    action = noop_action()
    action["camera"] = camera
    with pytest.raises(ValueError):
        validate_action(action)


def test_conflicting_buttons_and_inventory():
    codec = VPTActionCodec()
    action = noop_action()
    for name in ("forward", "back", "sprint", "sneak", "hotbar.2", "hotbar.9"):
        action["buttons"][name] = 1
    action["camera"][:] = [100, -100]
    canonical = codec.decode(codec.encode(action))
    assert canonical["buttons"]["forward"] == canonical["buttons"]["back"] == 0
    assert canonical["buttons"]["sneak"] == canonical["buttons"]["hotbar.9"] == 1
    assert canonical["buttons"]["sprint"] == canonical["buttons"]["hotbar.2"] == 0
    np.testing.assert_allclose(canonical["camera"], [10, -10])
    action["buttons"]["inventory"] = 1
    assert codec.encode(action) == {"buttons": 8640, "camera": 60}


@pytest.mark.parametrize(
    "encoded",
    [
        {"buttons": -1, "camera": 60},
        {"buttons": 8641, "camera": 60},
        {"buttons": 0, "camera": 121},
        {"buttons": 0.0, "camera": 60},
    ],
)
def test_invalid_encoded_action_rejected(encoded):
    with pytest.raises(ValueError):
        VPTActionCodec().decode(encoded)


def test_quantized_controls_are_canonical():
    codec = VPTActionCodec()
    for camera in range(121):
        encoded = {"buttons": 1 if camera != 60 else 0, "camera": camera}
        assert codec.encode(codec.decode(encoded)) == encoded
    assert codec.encode(noop_action()) == {"buttons": 0, "camera": 60}
