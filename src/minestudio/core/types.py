"""Framework independent observation and action contracts."""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class Observation(TypedDict):
    """RGB image: (height, width, 3), uint8, range [0, 255]."""

    image: NDArray[np.uint8]


class MinecraftAction(TypedDict):
    """Complete binary controls and (pitch, yaw) deltas in degrees."""

    buttons: dict[str, int]
    camera: NDArray[np.float32]


Info = dict[str, object]
StepResult = tuple[Observation, float, bool, bool, Info]


@dataclass(frozen=True)
class ImageSize:
    """Image dimensions in explicit height, width order."""

    height: int
    width: int

    def __post_init__(self) -> None:
        for value in (self.height, self.width):
            if type(value) is not int or value <= 0:
                raise ValueError("Image dimensions must be positive integers")


def snapshot_observation(observation: Observation) -> Observation:
    """Validate and copy so reused backend buffers cannot alter past observations."""
    image = observation["image"]
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Observation.image must be a uint8 RGB array of shape (H, W, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Observation.image cannot be empty")
    return {"image": image.copy()}
