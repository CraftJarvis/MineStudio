"""One RGB resize and action encoding contract for VPT inference and training."""

from collections.abc import Sequence

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from minestudio.actions import VPTActionCodec
from minestudio.core import ImageSize, TrajectoryWindow
from minestudio.policies.batches import PolicyBatch


def resize_image(image: NDArray[np.uint8], size: ImageSize) -> NDArray[np.uint8]:
    if image.shape[:2] != (size.height, size.width):
        image = cv2.resize(image, (size.width, size.height), interpolation=cv2.INTER_LINEAR)
    return image.copy()


def prepare_batch(
    windows: Sequence[TrajectoryWindow], *, size: ImageSize, device: torch.device
) -> PolicyBatch:
    if not windows or len({len(w.actions) for w in windows}) != 1:
        raise ValueError("A batch requires nonempty, equal-length windows")
    codec = VPTActionCodec()
    images, encoded, masks = [], [], []
    for window in windows:
        images.append(
            np.stack([resize_image(im, size) for im in window.observations["image"][:-1]])
        )
        mask = window.valid_mask & window.action_mask & window.observation_mask["image"][:-1]
        # Masked labels need not be usable codec inputs.
        encoded.append(
            [
                codec.encode(a) if valid else {"buttons": 0, "camera": 60}
                for a, valid in zip(window.actions, mask, strict=True)
            ]
        )
        masks.append(mask)
        known = window.observation_mask["image"][:-1]
        if np.any(~known[:-1] & known[1:]):
            raise ValueError("Internal image gaps require separate windows/context resets")
    return PolicyBatch(
        inputs={"image": torch.from_numpy(np.stack(images)).to(device)},
        actions={
            k: torch.tensor(
                [[a[k] for a in row] for row in encoded], device=device, dtype=torch.long
            ).unsqueeze(-1)
            for k in ("buttons", "camera")
        },
        valid_mask=torch.from_numpy(np.stack(masks)).to(device),
        first_mask=torch.from_numpy(np.stack([w.first_mask for w in windows])).to(device),
    )
