import runpy
from importlib.resources import files
from pathlib import Path

import pytest

select_render_device = runpy.run_path(
    str(files("minestudio") / "simulator/minerl/env/gpu_utils.py")
)["select_render_device"]


class Driver:
    def __init__(self, *, count=2, error=0):
        self.count = count
        self.error = error
        self.selected = None

    def cuInit(self, flags):
        return (self.error,)

    def cuDeviceGetCount(self):
        return 0, self.count

    def cuDeviceGet(self, index):
        self.selected = index
        return 0, index

    def cuDeviceGetPCIBusId(self, length, device):
        return 0, b"0000:A1:00.0\x00"


def test_gpu_selection_preserves_visible_device_order(tmp_path):
    driver = Driver()
    node = tmp_path / "pci-0000:a1:00.0-card"
    node.touch()
    assert select_render_device(3, driver=driver, device_root=tmp_path) == str(node)
    assert driver.selected == 1


@pytest.mark.parametrize("driver", [Driver(count=0), Driver(error=1)])
def test_explicit_gpu_request_never_silently_falls_back(driver):
    with pytest.raises(RuntimeError):
        select_render_device(0, driver=driver, device_root=Path("/nonexistent"))


def test_missing_drm_mapping_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="no DRM node"):
        select_render_device(0, driver=Driver(), device_root=tmp_path)
