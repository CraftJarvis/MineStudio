"""Fine-tune a local VPT export on fixed, source-disjoint dataset splits."""

import argparse
from pathlib import Path

import numpy as np
import torch

from minestudio.data import TrajectoryDataset, TrajectorySubset
from minestudio.policies.vpt import VPTPolicy
from minestudio.training import BCConfig, train_bc


def select_windows(dataset: TrajectoryDataset, per_record: int) -> TrajectorySubset:
    groups: dict[int, list[int]] = {}
    for index in range(len(dataset)):
        groups.setdefault(dataset.location(index)[0], []).append(index)
    indices = []
    for group in groups.values():
        positions = np.linspace(0, len(group) - 1, min(per_record, len(group)), dtype=int)
        indices.extend(group[i] for i in positions)
    return TrajectorySubset(dataset, indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="minestudio.dataset schema v1 manifest")
    parser.add_argument("policy", type=Path, help="Verified local VPT inference export")
    parser.add_argument("output", type=Path, help="A new directory, including when resuming")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--windows-per-record", type=int, default=8)
    parser.add_argument("--validation-windows-per-record", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.windows_per_record < 1 or args.validation_windows_per_record < 1:
        parser.error("Window counts must be positive")
    torch.set_num_threads(4)
    policy = VPTPolicy.from_pretrained(args.policy, device=args.device)
    with (
        TrajectoryDataset(
            args.dataset, sequence_length=args.sequence_length, split="train", pad_end=False
        ) as train,
        TrajectoryDataset(
            args.dataset, sequence_length=args.sequence_length, split="validation", pad_end=False
        ) as validation,
    ):
        result = train_bc(
            policy,
            select_windows(train, args.windows_per_record),
            validation_dataset=select_windows(validation, args.validation_windows_per_record),
            config=BCConfig(
                output_dir=args.output,
                max_steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                precision=args.precision,
                seed=args.seed,
            ),
            resume_from=args.resume,
        )
    print(f"Export: {result.export}\nCheckpoint: {result.checkpoint}\nMetrics: {result.metrics}")


if __name__ == "__main__":
    main()
