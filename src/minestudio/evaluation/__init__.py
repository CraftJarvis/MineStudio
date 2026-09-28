"""Task-aware evaluation without importing a concrete environment or model."""

from minestudio.evaluation.api import EvaluationEnvFactory, evaluate
from minestudio.evaluation.config import EvaluationConfig, EvaluationSuite
from minestudio.evaluation.results import EvaluationResult

__all__ = [
    "EvaluationConfig",
    "EvaluationEnvFactory",
    "EvaluationResult",
    "EvaluationSuite",
    "evaluate",
]
