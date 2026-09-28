"""Lightweight policy contracts with lazy model exports."""

from typing import TYPE_CHECKING

from minestudio.policies.protocols import Policy

if TYPE_CHECKING:
    from minestudio.policies.vpt.policy import VPTPolicy as VPTPolicy
__all__ = ["Policy", "VPTPolicy"]


def __getattr__(name: str) -> object:
    if name == "VPTPolicy":
        from minestudio.policies.vpt.policy import VPTPolicy

        return VPTPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
