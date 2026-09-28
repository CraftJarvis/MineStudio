"""Transitional v1 exports; imported only when explicitly requested."""
from importlib import import_module
_EXPORTS = {'MinePolicy': 'base_policy', 'VPTPolicy': 'vpt', 'load_vpt_policy': 'vpt', 'GrootPolicy': 'groot_one', 'load_groot_policy': 'groot_one', 'RocketPolicy': 'rocket_one', 'load_rocket_policy': 'rocket_one', 'SteveOnePolicy': 'steve_one', 'load_steve_one_policy': 'steve_one'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f"minestudio.models.{_EXPORTS[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
