"""Dependency/mission construction smoke check; does not start Minecraft."""

import os
import subprocess
import sys

import pytest


def test_minerl_import_without_training_frameworks(tmp_path):
    for module in ("gym", "cv2", "rich", "diskcache", "Pyro4", "lxml", "daemoniker"):
        pytest.importorskip(module)
    script = """
import importlib.abc
import sys
import numpy as np
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'ray', 'lightning'}:
            raise AssertionError(f'Unexpected training import: {fullname}')
sys.meta_path.insert(0, RejectTraining())
from minestudio.actions import BUTTONS
from minestudio.simulator.minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
rng = np.random.RandomState(12)
state = np.random.get_state()
spec = HumanSurvival(resolution=(32, 24), rng=rng)
assert set(BUTTONS).issubset(spec.action_space.spaces)
xml = spec.to_xml()
assert 'Mission' in xml
env = spec.make(is_fault_tolerant=False)
assert not env.instances
env.close()
assert np.array_equal(np.random.get_state()[1], state[1])
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=dict(os.environ, MINESTUDIO_DIR=str(tmp_path)),
    )
