"""NumPy-only VPT hierarchical encoding compatible with the original action map."""

from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from minestudio.actions.minecraft import noop_action, validate_action
from minestudio.core import MinecraftAction

_GROUPS = (
    tuple(f"hotbar.{i}" for i in range(1, 10)),
    ("forward", "back"),
    ("left", "right"),
    ("sprint", "sneak"),
    ("use",),
    ("drop",),
    ("attack",),
    ("jump",),
)
_RADICES = (10, 3, 3, 3, 2, 2, 2, 2, 2)


class VPTAction(TypedDict):
    """Scalar button index [0, 8641) and camera index [0, 121)."""

    buttons: int
    camera: int


class VPTActionCodec:
    """Fixed mu-law camera codec: max 10 degrees, mu 10, 11 bins per axis.

    Opposing directions cancel; sneak and higher hotbar slots win. Encoding
    inventory suppresses other controls. Decoding preserves the original
    inventory-camera behavior. Quantization and control resolution are lossy.
    """

    name = "vpt-mu-law-v1"
    num_buttons = 8641
    num_camera = 121

    def encode(self, action: MinecraftAction) -> VPTAction:
        validate_action(action)
        if action["buttons"]["inventory"]:
            return {"buttons": 8640, "camera": 60}
        camera = np.clip(action["camera"].astype(np.float64), -10, 10)
        scaled = cast(
            NDArray[np.float64], np.sign(camera) * np.log1p(np.abs(camera)) / np.log(11.0) * 10
        )
        bins = np.round((scaled + 10) / 2).astype(np.int64)
        choices: list[int] = []
        for group in _GROUPS:
            active = [i + 1 for i, name in enumerate(group) if action["buttons"][name]]
            choice = active[-1] if active else 0
            if group in (("forward", "back"), ("left", "right")) and len(active) == 2:
                choice = 0
            choices.append(choice)
        choices.append(int(np.any(bins != 5)))
        index = 0
        for choice, radix in zip(choices, _RADICES, strict=True):
            index = index * radix + choice
        return {"buttons": index, "camera": int(bins[0] * 11 + bins[1])}

    def decode(self, policy_action: VPTAction) -> MinecraftAction:
        if set(policy_action) != {"buttons", "camera"}:
            raise ValueError("VPTAction requires buttons and camera")
        for key, value, limit in (
            ("buttons", policy_action["buttons"], self.num_buttons),
            ("camera", policy_action["camera"], self.num_camera),
        ):
            if type(value) is not int or not 0 <= value < limit:
                raise ValueError(f"VPT {key} must be an integer in [0, {limit})")
        index = policy_action["buttons"]
        action = noop_action()
        if index == 8640:
            action["buttons"]["inventory"] = 1
            camera_on = True
        else:
            choices: list[int] = []
            for radix in reversed(_RADICES):
                choices.append(index % radix)
                index //= radix
            choices.reverse()
            for group, choice in zip(_GROUPS, choices[:-1], strict=True):
                if choice:
                    action["buttons"][group[choice - 1]] = 1
            camera_on = bool(choices[-1])
        if camera_on:
            bins = np.array(divmod(policy_action["camera"], 11), dtype=np.float64)
            scaled = (bins * 2 - 10) / 10
            action["camera"] = (np.sign(scaled) * np.expm1(np.abs(scaled) * np.log(11))).astype(
                np.float32
            )
        return action
