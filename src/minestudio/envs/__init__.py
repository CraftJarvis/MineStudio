"""Environment contracts and lazy Gymnasium entry point."""

from typing import TYPE_CHECKING

from minestudio.envs.config import EnvConfig
from minestudio.envs.protocols import EnvBackend, Environment

if TYPE_CHECKING:
    from minestudio.envs.environment import MinecraftEnv as MinecraftEnv
__all__ = ["EnvBackend", "EnvConfig", "Environment", "MinecraftEnv"]


def __getattr__(name: str) -> object:
    if name == "MinecraftEnv":
        from minestudio.envs.environment import MinecraftEnv

        return MinecraftEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
