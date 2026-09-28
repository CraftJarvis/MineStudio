"""VPT configuration and adapter; runtime dependencies load on policy access."""

from typing import TYPE_CHECKING

from minestudio.policies.vpt.config import VPTConfig

if TYPE_CHECKING:
    from minestudio.policies.vpt.policy import VPTPolicy as VPTPolicy
__all__ = ["VPTConfig", "VPTPolicy"]


def __getattr__(name: str) -> object:
    if name == "VPTPolicy":
        from minestudio.policies.vpt.policy import VPTPolicy

        return VPTPolicy
    raise AttributeError(name)
