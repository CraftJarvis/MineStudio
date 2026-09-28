"""Resolve a visible CUDA device to VirtualGL's DRM device path.

CPU rendering is the default. An explicit GPU request must either select a GPU
or fail visibly; it must never silently fall back to software rendering.
"""

import argparse
import os
from pathlib import Path


def _driver():
    try:
        from cuda.bindings import driver
    except ImportError:
        try:
            from cuda import cuda as driver
        except ImportError as error:
            raise RuntimeError("GPU rendering requires cuda-bindings or cuda-python") from error
    return driver


def _checked(result):
    if result[0] != 0:
        raise RuntimeError(f"CUDA driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else None


def select_render_device(
    index: int, *, driver=None, device_root: Path = Path("/dev/dri/by-path")
) -> str:
    """Select among CUDA_VISIBLE_DEVICES and verify the corresponding DRM node."""
    if type(index) is not int or index < 0:
        raise ValueError("Render device index must be a nonnegative integer")
    driver = _driver() if driver is None else driver
    _checked(driver.cuInit(0))
    count = _checked(driver.cuDeviceGetCount())
    if not count:
        raise RuntimeError("GPU rendering requested but no CUDA devices are visible")
    device = _checked(driver.cuDeviceGet(index % count))
    bus_id = _checked(driver.cuDeviceGetPCIBusId(100, device))
    if isinstance(bus_id, bytes):
        bus_id = bus_id.decode("ascii")
    bus_id = bus_id.split("\0", 1)[0].lower()
    path = device_root / f"pci-{bus_id}-card"
    if not path.exists():
        raise RuntimeError(f"CUDA device has no DRM node for VirtualGL: {path}")
    if not os.access(path, os.R_OK | os.W_OK):
        raise RuntimeError(f"DRM node is not accessible: {path}")
    return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    if os.environ.get("MINESTUDIO_GPU_RENDER") != "1":
        print("cpu")
        return
    try:
        print(select_render_device(args.index))
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f"GPU rendering unavailable: {error}\n")


if __name__ == "__main__":
    main()
