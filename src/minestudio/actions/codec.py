"""Minimal action codec protocol."""

from typing import Protocol, TypeVar

from minestudio.core import MinecraftAction

EncodedT = TypeVar("EncodedT")


class ActionCodec(Protocol[EncodedT]):
    """Translate native actions and a documented, potentially lossy encoding."""

    def encode(self, action: MinecraftAction) -> EncodedT: ...
    def decode(self, policy_action: EncodedT) -> MinecraftAction: ...
