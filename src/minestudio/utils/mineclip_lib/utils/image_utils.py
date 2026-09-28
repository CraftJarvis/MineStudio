from __future__ import annotations

import warnings

import torch

from .torch_utils import torch_normalize


@torch.no_grad()
def basic_image_tensor_preprocess(
    img,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    shape: tuple[int, int] | None = None,
):
    """
    Check for resize, and divide by 255
    """
   
    assert torch.is_tensor(img)
    assert img.dim() >= 4
    original_shape = list(img.size())
    img = img.float()
    img = img.flatten(0, img.dim() - 4)
    assert img.dim() == 4

    input_size = img.size()[-2:]

    if shape and input_size != shape:
        import kornia
        warnings.warn(
            f'{"Down" if shape < input_size else "Up"}sampling image'
            f" from original resolution {input_size}x{input_size}"
            f" to {shape}x{shape}"
        )
        img = kornia.geometry.transform.resize(img, shape).clamp(0.0, 255.0)

    B, C, H, W = img.size()
    assert C % 3 == 0, "channel must divide 3"
    img = img.view(B * C // 3, 3, H, W)
    img = torch_normalize(img / 255.0, mean=mean, std=std)
    original_shape[-2:] = H, W
    return img.view(original_shape)
