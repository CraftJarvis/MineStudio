"""Optional single-device PyTorch BC and synchronous PPO."""

from minestudio.training.bc import BCConfig, TrainingResult, train_bc
from minestudio.training.ppo import PPOTrainingResult, train_ppo
from minestudio.training.ppo_config import PPOConfig

__all__ = ["BCConfig", "PPOConfig", "PPOTrainingResult", "TrainingResult", "train_bc", "train_ppo"]
