import os
import subprocess
import sys


def test_lightweight_imports_without_optional_modules():
    script = """
import importlib.abc
import sys
class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'ray', 'lightning', 'cv2', 'gymnasium', 'gym'}:
            raise AssertionError(f'unexpected eager import: {fullname}')
sys.meta_path.insert(0, RejectOptional())
import minestudio
from minestudio.core import Observation
from minestudio.actions import VPTActionCodec
from minestudio.policies import Policy
from minestudio.envs import EnvConfig
from minestudio.rollout import RolloutRunner
from minestudio.data import TrajectoryReader
from minestudio.tasks import TaskSpec
from minestudio.evaluation import evaluate
assert minestudio.__version__ == '2.0.0a2'
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_doctor_does_not_launch_engine(tmp_path):
    env = dict(os.environ, MINESTUDIO_DIR=str(tmp_path / "absent"))
    result = subprocess.run(
        [sys.executable, "-m", "minestudio.cli.main", "doctor"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert '"engine_present": false' in result.stdout
    assert not (tmp_path / "absent").exists()
