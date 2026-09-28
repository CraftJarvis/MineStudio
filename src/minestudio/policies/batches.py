"""Differentiable policy capability, independent of training backends."""

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from minestudio.core import MinecraftAction, Observation


@dataclass(frozen=True)
class PolicyBatch:
    """Encoded [B,T,...] inputs with a combined BC supervision mask.

    valid_mask combines real steps, current-image availability and action labels.
    first_mask retains known episode starts; an empty state is a separate decision.
    """

    inputs: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor]
    valid_mask: torch.Tensor
    first_mask: torch.Tensor


@dataclass(frozen=True)
class ActionEvaluation:
    """Per-step differentiable log probabilities and caller-owned next state."""

    log_prob: torch.Tensor
    head_log_probs: dict[str, torch.Tensor]
    entropy: torch.Tensor
    next_state: Any


@dataclass(frozen=True)
class ActorCriticEvaluation(ActionEvaluation):
    """Value is differentiable and expressed in the environment's reward units."""

    value: torch.Tensor


class LikelihoodPolicy(Protocol):
    def evaluate_actions(self, batch: PolicyBatch, state: Any = None) -> ActionEvaluation: ...


class ActorCriticStep(Protocol):
    """Statistics for the exact encoded sample taken by the behavior policy."""

    @property
    def action(self) -> MinecraftAction: ...
    @property
    def policy_action(self) -> Any: ...
    @property
    def log_prob(self) -> float: ...
    @property
    def value(self) -> float: ...


class ActorCriticPolicy(LikelihoodPolicy, Protocol):
    def act_with_stats(
        self,
        observation: Observation,
        state: Any = None,
        *,
        deterministic: bool = False,
    ) -> tuple[ActorCriticStep, Any]: ...
    def value(self, observation: Observation, state: Any = None) -> float: ...
    def evaluate_actions(self, batch: PolicyBatch, state: Any = None) -> ActorCriticEvaluation: ...
