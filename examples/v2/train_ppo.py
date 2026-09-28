"""Train synchronous VPT PPO on a fixed Minecraft task using local factories."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from minestudio.envs import EnvConfig, MinecraftEnv
from minestudio.policies.vpt import VPTPolicy
from minestudio.tasks import load_task
from minestudio.training import PPOConfig, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, help="Verified local VPT export")
    parser.add_argument("output", type=Path, help="A new output directory")
    parser.add_argument("--task", default="collect_oak_log", help="Built-in task or JSON/YAML path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-sequences", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--save-rollouts", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    task = load_task(args.task)
    env_config = EnvConfig()
    environment = {"task": task.to_dict(), "env": asdict(env_config), "backend": "minerl"}
    environment_id = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(4)
    result = train_ppo(
        lambda: VPTPolicy.from_pretrained(args.policy, device=args.device),
        lambda: MinecraftEnv(env_config, task=task),
        config=PPOConfig(
            output_dir=args.output,
            environment_id=environment_id,
            iterations=args.iterations,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            sequence_length=args.sequence_length,
            minibatch_sequences=args.minibatch_sequences,
            update_epochs=args.update_epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            env_seed=args.env_seed,
            save_rollouts=args.save_rollouts,
        ),
        resume_from=args.resume,
    )
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(f"Export: {result.export}\nCheckpoint: {result.checkpoint}\nMetrics: {result.metrics}")


if __name__ == "__main__":
    main()
