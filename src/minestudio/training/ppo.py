"""Single-device synchronous PPO with explicit recurrent and restart semantics."""

import json
import random
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from minestudio.envs.protocols import Environment
from minestudio.policies.batches import ActorCriticEvaluation
from minestudio.training._checkpoint import restore_rng, rng_state, tensor_tree
from minestudio.training.ppo_config import PPOConfig
from minestudio.training.ppo_objective import ppo_loss
from minestudio.training.ppo_rollout import PPORollout, batch_to, collect_ppo_rollout


@dataclass(frozen=True)
class PPOTrainingResult:
    run_id: str
    iterations: int
    optimizer_steps: int
    environment_steps: int
    config_path: Path
    checkpoint: Path
    export: Path
    metrics: Path


@torch.no_grad()
def check_behavior(policy: Any, rollout: PPORollout, tolerance: float) -> dict[str, float]:
    """Gate every update on sampled-code likelihood and value reconstruction."""
    log_error, value_error, ratio_error = 0.0, 0.0, 0.0
    for sequence in rollout.sequences:
        if (
            sequence.policy_version != rollout.policy_version
            or sequence.state_version != rollout.policy_version
        ):
            raise ValueError("Mixed policy/cache versions in PPO rollout")
        batch = batch_to(sequence.batch, policy.device)
        evaluation = policy.evaluate_actions(
            batch, tensor_tree(sequence.initial_state, policy.device)
        )
        if not isinstance(evaluation, ActorCriticEvaluation):
            raise TypeError("PPO requires ActorCriticEvaluation, not a BC-only likelihood policy")
        old_log_prob, old_value = (
            sequence.old_log_prob.to(policy.device),
            sequence.old_value.to(policy.device),
        )
        if (
            not torch.isfinite(evaluation.log_prob).all()
            or not torch.isfinite(evaluation.value).all()
        ):
            raise FloatingPointError("Non-finite behavior reconstruction")
        error = evaluation.log_prob - old_log_prob
        log_error = max(log_error, float(error.abs().max()))
        ratio_error = max(ratio_error, float((error.exp() - 1).abs().max()))
        relative_value_error = float(
            ((evaluation.value - old_value).abs() / old_value.abs().clamp(min=1)).max()
        )
        value_error = max(value_error, relative_value_error)
    if log_error > tolerance or value_error > tolerance:
        raise ValueError(
            f"Behavior reconstruction failed: log_prob={log_error}, relative_value={value_error}"
        )
    return {
        "behavior_log_prob_error": log_error,
        "behavior_value_error": value_error,
        "initial_ratio_error": ratio_error,
    }


def train_ppo(
    policy_factory: Callable[[], Any],
    env_factory: Callable[[], Environment],
    *,
    config: PPOConfig,
    resume_from: str | Path | None = None,
) -> PPOTrainingResult:
    """Own factory-created resources; collect fresh rounds then optimize shuffled sequences.

    The model stays in eval mode while gradients are enabled to preserve sampling
    distributions and freeze value normalizer statistics. Losses use reward units.
    Checkpoints are written after whole rounds. Resume resets environments with the
    next saved seed; it never claims to restore a live Minecraft world.
    """
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(output)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    policy = policy_factory().eval()
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in policy.parameters()):
        raise ValueError("This PPO implementation requires FP32 policy parameters")
    for method in (
        "act_with_stats",
        "value",
        "prepare_sequence",
        "batch_state",
        "evaluate_actions",
    ):
        if not callable(getattr(policy, method, None)):
            raise TypeError(f"PPO policy requires {method}")
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=0, eps=1e-5
    )
    generator = torch.Generator().manual_seed(config.seed)
    identity = {
        "config": {
            k: v
            for k, v in asdict(config).items()
            if k not in ("output_dir", "iterations", "save_rollouts")
        },
        "model_config": asdict(policy.config),
        "codec": policy.codec.name,
        "processor": "rgb-uint8-cv2-linear-v1",
        "parameters": [(n, tuple(p.shape), p.requires_grad) for n, p in policy.named_parameters()],
        "context": "reset_each_round; detached_behavior_cache_per_sequence",
        "value_normalization": "frozen; losses_in_reward_units",
        "precision": "float32",
    }
    iteration, optimizer_steps, environment_steps, next_episode = 0, 0, 0, 0
    initial_policy = getattr(policy, "_pretrained_manifest", None)
    restored_rng = None
    if resume_from is not None:
        saved = torch.load(resume_from, map_location="cpu", weights_only=True)
        if saved.get("format") != "minestudio.ppo-checkpoint" or saved.get("schema_version") != 1:
            raise ValueError("Unsupported PPO checkpoint")
        if saved["identity"] != identity:
            raise ValueError("PPO checkpoint model, environment or objective differs")
        iteration = saved["iteration"]
        if iteration >= config.iterations:
            raise ValueError("iterations must exceed the completed checkpoint round")
        policy.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        optimizer_steps, environment_steps = saved["optimizer_steps"], saved["environment_steps"]
        next_episode, initial_policy = saved["next_episode"], saved["initial_policy"]
        generator.set_state(saved["sampler_rng"])
        restored_rng = saved["rng"]
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    config_path, metrics, checkpoint = (
        output / "config.json",
        output / "metrics.jsonl",
        output / "checkpoint.pt",
    )
    config_path.write_text(
        json.dumps(
            {
                **asdict(config),
                "output_dir": str(output),
                "identity": identity,
                "run_id": run_id,
                "initial_policy": initial_policy,
            },
            indent=2,
        )
        + "\n"
    )
    start_time = time.monotonic()

    def log(row):
        row = {
            "iteration": iteration,
            "optimizer_steps": optimizer_steps,
            "environment_steps": environment_steps,
            "elapsed_seconds": time.monotonic() - start_time,
            **row,
        }
        with metrics.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    with ExitStack() as stack:
        environments = []
        for _ in range(config.num_envs):
            env = env_factory()
            stack.callback(env.close)
            environments.append(env)
        if restored_rng is not None:
            restore_rng(restored_rng)
        while iteration < config.iterations:
            rollout = collect_ppo_rollout(
                policy,
                environments,
                config=config,
                policy_version=iteration,
                next_episode=next_episode,
            )
            next_episode = rollout.next_episode
            environment_steps += config.num_envs * config.rollout_steps
            log(
                {
                    "phase": "collected",
                    **check_behavior(policy, rollout, config.behavior_tolerance),
                    "reward_sum": sum(float(s.rewards.sum()) for s in rollout.sequences),
                    "episodes": rollout.episodes,
                }
            )
            if config.save_rollouts:
                torch.save(rollout.payload(), output / f"rollout-{iteration:06d}.pt")
            early_stop = False
            for epoch in range(config.update_epochs):
                order = torch.randperm(len(rollout.sequences), generator=generator).tolist()
                for start in range(0, len(order), config.minibatch_sequences):
                    selected = [
                        rollout.sequences[i]
                        for i in order[start : start + config.minibatch_sequences]
                    ]
                    total_tokens = sum(int(s.batch.valid_mask.sum()) for s in selected)
                    optimizer.zero_grad(set_to_none=True)
                    stats = dict.fromkeys(
                        (
                            "loss",
                            "policy_loss",
                            "value_loss",
                            "entropy",
                            "approx_kl",
                            "clip_fraction",
                        ),
                        0.0,
                    )
                    for sequence in selected:
                        batch = batch_to(sequence.batch, policy.device)
                        evaluation = policy.evaluate_actions(
                            batch, tensor_tree(sequence.initial_state, policy.device)
                        )
                        losses = ppo_loss(
                            evaluation,
                            batch,
                            old_log_prob=sequence.old_log_prob.to(policy.device),
                            old_value=sequence.old_value.to(policy.device),
                            advantages=sequence.advantages.to(policy.device),
                            returns=sequence.returns.to(policy.device),
                            config=config,
                        )
                        weight = int(batch.valid_mask.sum()) / total_tokens
                        (losses.loss * weight).backward()
                        for key in stats:
                            stats[key] += float(getattr(losses, key).detach()) * weight
                    if config.target_kl is not None and stats["approx_kl"] > config.target_kl:
                        optimizer.zero_grad(set_to_none=True)
                        early_stop = True
                        log({"phase": "kl_stop", "epoch": epoch, **stats})
                        break
                    norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), config.max_grad_norm, error_if_nonfinite=True
                    )
                    optimizer.step()
                    optimizer_steps += 1
                    log({"phase": "updated", "epoch": epoch, "grad_norm": float(norm), **stats})
                if early_stop:
                    break
            iteration += 1
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(
                {
                    "format": "minestudio.ppo-checkpoint",
                    "schema_version": 1,
                    "identity": identity,
                    "initial_policy": initial_policy,
                    "iteration": iteration,
                    "optimizer_steps": optimizer_steps,
                    "environment_steps": environment_steps,
                    "next_episode": next_episode,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sampler_rng": generator.get_state(),
                    "rng": rng_state(),
                    "environment_state": None,
                    "recurrent_state": None,
                    "scheduler": None,
                },
                temporary,
            )
            temporary.replace(checkpoint)
    export = output / "policy"
    policy.save_pretrained(
        export,
        provenance={
            "format": "minestudio.ppo",
            "iterations": iteration,
            "environment_steps": environment_steps,
            "environment_id": config.environment_id,
            "initial_policy": initial_policy,
            "objective": identity["config"],
        },
    )
    return PPOTrainingResult(
        run_id,
        iteration,
        optimizer_steps,
        environment_steps,
        config_path,
        checkpoint,
        export,
        metrics,
    )
