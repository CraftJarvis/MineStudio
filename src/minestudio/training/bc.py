"""BC over independent windows, with exact optimizer-boundary checkpoint recovery."""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

import numpy as np
import torch

from minestudio.core import TrajectoryWindow
from minestudio.policies.batches import ActionEvaluation, PolicyBatch
from minestudio.training._checkpoint import restore_rng as _restore_rng
from minestudio.training._checkpoint import rng_state as _rng_state


class WindowDataset(Protocol):
    fingerprint: str

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> TrajectoryWindow: ...


@dataclass(frozen=True)
class BCConfig:
    """An optimizer step supervises batch_size windows, each with empty context.

    Loss is the mean over valid labelled steps of weighted per-head NLL.
    Camera no-op labels are supervised by default. Setting mask_neutral_camera
    reproduces v1's head masking, but deliberately retains valid-step normalization.
    This initial trainer has a constant LR and no cross-window state or scheduler.
    """

    output_dir: str | Path
    max_steps: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    buttons_weight: float = 1.0
    camera_weight: float = 1.0
    mask_neutral_camera: bool = False
    precision: Literal["float32", "bfloat16"] = "float32"
    seed: int = 0
    validation_interval: int = 20
    checkpoint_interval: int = 50
    context_mode: Literal["independent_windows"] = "independent_windows"

    def __post_init__(self) -> None:
        for key in ("max_steps", "batch_size", "validation_interval", "checkpoint_interval"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in (
            "learning_rate",
            "max_grad_norm",
            "weight_decay",
            "buttons_weight",
            "camera_weight",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        if self.learning_rate == 0 or self.max_grad_norm == 0:
            raise ValueError("Learning rate and gradient norm limit must be positive")
        if self.buttons_weight + self.camera_weight == 0:
            raise ValueError("At least one head needs a positive weight")
        if self.context_mode != "independent_windows" or self.precision not in (
            "float32",
            "bfloat16",
        ):
            raise ValueError("Unsupported context or precision mode")


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    steps: int
    config_path: Path
    checkpoint: Path
    export: Path
    metrics: Path
    final_loss: float
    final_validation_loss: float | None


def bc_loss(evaluation: ActionEvaluation, batch: PolicyBatch, config: BCConfig) -> torch.Tensor:
    """Explicit token-normalized objective; missing labels and padding contribute zero."""
    mask = batch.valid_mask
    if not torch.any(mask):
        raise ValueError("BC batch has no supervised steps")
    buttons = evaluation.head_log_probs["buttons"].float()
    camera = evaluation.head_log_probs["camera"].float()
    if buttons.shape != mask.shape or camera.shape != mask.shape:
        raise ValueError("Log probabilities and supervision mask must be [B,T]")
    if config.mask_neutral_camera:
        camera = torch.where(batch.actions["camera"].squeeze(-1) != 60, camera, 0.0)
    loss = -(config.buttons_weight * buttons + config.camera_weight * camera)
    return loss[mask].mean()


def _config_identity(config: BCConfig) -> dict:
    return {
        k: v
        for k, v in asdict(config).items()
        if k not in ("output_dir", "max_steps", "validation_interval", "checkpoint_interval")
    }


@torch.no_grad()
def validation_loss(policy: Any, dataset: WindowDataset, config: BCConfig) -> float:
    """Evaluate all supplied windows with the same empty-context objective."""
    was_training = policy.training
    total, count = 0.0, 0
    policy.eval()
    try:
        for start in range(0, len(dataset), config.batch_size):
            batch = policy.prepare_batch(
                [dataset[i] for i in range(start, min(start + config.batch_size, len(dataset)))]
            )
            tokens = int(batch.valid_mask.sum())
            if tokens == 0:
                continue
            with torch.autocast(
                policy.device.type, dtype=torch.bfloat16, enabled=config.precision == "bfloat16"
            ):
                loss = bc_loss(policy.evaluate_actions(batch), batch, config)
            total += float(loss) * tokens
            count += tokens
    finally:
        policy.train(was_training)
    if count == 0:
        raise ValueError("Validation dataset has no supervised steps")
    return total / count


def train_bc(
    policy: Any,
    dataset: WindowDataset,
    *,
    config: BCConfig,
    validation_dataset: WindowDataset | None = None,
    resume_from: str | Path | None = None,
) -> TrainingResult:
    """Train a caller-owned likelihood policy; keep caller-owned datasets open.

    Checkpoints capture model/optimizer, RNG and shuffled index order/cursor. Resume
    creates a new output directory and requires identical data, policy and objective.
    max_steps is the absolute optimizer step to reach. No partial-step resume.
    """
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")
    if config.precision == "bfloat16" and (
        policy.device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("bfloat16 training requires a supported CUDA device")
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    step, cursor, order = 0, 0, []
    initial_policy = getattr(policy, "_pretrained_manifest", None)
    model_identity = {
        "config": asdict(policy.config),
        "codec": policy.codec.name,
        "processor": "rgb-uint8-cv2-linear-v1",
        "parameters": [(n, tuple(p.shape), p.requires_grad) for n, p in policy.named_parameters()],
    }
    identity = {
        "config": _config_identity(config),
        "dataset": dataset.fingerprint,
        "validation_dataset": None
        if validation_dataset is None
        else validation_dataset.fingerprint,
        "model": model_identity,
    }
    if resume_from is not None:
        saved = torch.load(resume_from, map_location="cpu", weights_only=True)
        if saved.get("format") != "minestudio.bc-checkpoint" or saved.get("schema_version") != 1:
            raise ValueError("Unsupported BC checkpoint")
        if saved["identity"] != identity:
            raise ValueError("Checkpoint data, model or objective differs from this run")
        step = saved["step"]
        initial_policy = saved.get("initial_policy")
        if step >= config.max_steps:
            raise ValueError("max_steps must exceed the checkpoint step")
        policy.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        order, cursor = saved["order"], saved["cursor"]
        generator.set_state(saved["sampler_rng"])
        _restore_rng(saved["rng"])
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    checkpoint, metrics = output / "checkpoint.pt", output / "metrics.jsonl"
    (output / "config.json").write_text(
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
        row = {"step": step, "elapsed_seconds": time.monotonic() - start_time, **row}
        with metrics.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    def save_checkpoint():
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "format": "minestudio.bc-checkpoint",
                "schema_version": 1,
                "identity": identity,
                "initial_policy": initial_policy,
                "step": step,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "order": order,
                "cursor": cursor,
                "sampler_rng": generator.get_state(),
                "rng": _rng_state(),
                "recurrent_state": None,
                "scheduler": None,
            },
            temporary,
        )
        temporary.replace(checkpoint)

    was_training = policy.training
    final_validation_loss = None
    try:
        if validation_dataset is not None:
            log({"validation_loss": validation_loss(policy, validation_dataset, config)})
        policy.train()
        while step < config.max_steps:
            indices = []
            for _ in range(config.batch_size):
                if cursor == len(order):
                    order, cursor = torch.randperm(len(dataset), generator=generator).tolist(), 0
                indices.append(order[cursor])
                cursor += 1
            batch = policy.prepare_batch([dataset[i] for i in indices])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                policy.device.type, dtype=torch.bfloat16, enabled=config.precision == "bfloat16"
            ):
                loss = bc_loss(policy.evaluate_actions(batch), batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite BC loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            log(
                {
                    "loss": float(loss.detach()),
                    "grad_norm": float(norm),
                    "supervised_steps": int(batch.valid_mask.sum()),
                    "indices": indices,
                }
            )
            if validation_dataset is not None and (
                step % config.validation_interval == 0 or step == config.max_steps
            ):
                final_validation_loss = validation_loss(policy, validation_dataset, config)
                log({"validation_loss": final_validation_loss})
            if step % config.checkpoint_interval == 0 or step == config.max_steps:
                save_checkpoint()
        export = output / "policy"
        policy.save_pretrained(
            export,
            provenance={
                "format": "minestudio.bc",
                "step": step,
                "dataset": dataset.fingerprint,
                "objective": _config_identity(config),
                "initial_policy": initial_policy,
            },
        )
        return TrainingResult(
            run_id,
            step,
            output / "config.json",
            checkpoint,
            export,
            metrics,
            float(loss.detach()),
            final_validation_loss,
        )
    finally:
        policy.train(was_training)
