"""Transitional v1 exports; imported only when explicitly requested."""
from importlib import import_module
_EXPORTS = {'RawDataset': 'minecraft', 'RawDataModule': 'minecraft', 'EventDataset': 'minecraft', 'EventDataModule': 'minecraft'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f"minestudio.data.{_EXPORTS[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(name)

from minestudio.data.storage import RecordedTrajectory, TrajectoryReader, TrajectoryWriter
__all__ += ["RecordedTrajectory", "TrajectoryReader", "TrajectoryWriter"]
