"""Native Minecraft actions, independent of policy encoding."""

import numpy as np

from minestudio.core import MinecraftAction

BUTTONS = (
    "attack",
    "back",
    "forward",
    "jump",
    "left",
    "right",
    "sneak",
    "sprint",
    "use",
    "drop",
    "inventory",
    *(f"hotbar.{index}" for index in range(1, 10)),
)


def noop_action() -> MinecraftAction:
    """Create a fresh neutral action."""
    return {"buttons": dict.fromkeys(BUTTONS, 0), "camera": np.zeros(2, dtype=np.float32)}


def validate_action(action: MinecraftAction) -> None:
    """Reject incomplete controls, invalid values, and malformed camera arrays."""
    if set(action) != {"buttons", "camera"} or set(action["buttons"]) != set(BUTTONS):
        raise ValueError("Action requires buttons (the complete BUTTONS schema) and camera")
    if any(
        not isinstance(value, (int, np.integer)) or value not in (0, 1)
        for value in action["buttons"].values()
    ):
        raise ValueError("Button values must be integers 0 or 1")
    camera = action["camera"]
    if camera.dtype != np.float32 or camera.shape != (2,) or not np.isfinite(camera).all():
        raise ValueError("Camera must be a finite float32 array of shape (2,)")


def copy_action(action: MinecraftAction) -> MinecraftAction:
    """Validate and copy before handing an action to a mutable backend."""
    validate_action(action)
    return {
        "buttons": {name: int(value) for name, value in action["buttons"].items()},
        "camera": action["camera"].copy(),
    }
