"""Explicit task evaluation and safe v1 state-dict conversion commands."""

import argparse
import json
from pathlib import Path

from minestudio.data.storage.trajectory import sha256_file
from minestudio.envs.protocols import Environment
from minestudio.tasks import TaskSpec, load_task


def read_task(source: str) -> TaskSpec:
    path = Path(source)
    if path.is_file():
        return (
            TaskSpec.from_yaml(path)
            if path.suffix in {".yaml", ".yml"}
            else TaskSpec.from_json(path)
        )
    return load_task(source)


def _minecraft_task_env(*, task: TaskSpec) -> "Environment":
    from minestudio.envs import MinecraftEnv

    return MinecraftEnv(task=task)


def run_evaluation(args: argparse.Namespace) -> None:
    from minestudio.envs.engine import require_engine
    from minestudio.evaluation import EvaluationConfig, EvaluationSuite, evaluate
    from minestudio.policies.vpt import VPTPolicy

    task = read_task(args.task)
    require_engine()
    export = Path(args.policy)
    manifest = json.loads((export / "manifest.json").read_text())
    result = evaluate(
        lambda: VPTPolicy.from_pretrained(export, device=args.device),
        suite=EvaluationSuite(args.suite_id, "1", (task,), tuple(args.seeds), args.repeats),
        env_factory=_minecraft_task_env,
        config=EvaluationConfig(
            policy_id=str(export.resolve()),
            policy_revision=sha256_file(export / "manifest.json"),
            processor_revision=manifest["processor"],
            codec_revision=manifest["codec"],
            output_dir=args.output_dir,
            max_episode_steps=args.max_episode_steps,
            num_envs=args.num_envs,
            max_retries=args.max_retries,
            record_video=args.video,
        ),
    )
    print(
        json.dumps(
            {
                "completed": result.completed,
                "failed": result.failed,
                "success_rate": result.success_rate,
                "report": str(result.report_path),
            },
            indent=2,
        )
    )


def convert_vpt(args: argparse.Namespace) -> None:
    from minestudio.policies.vpt.conversion import convert_legacy_export

    print(convert_legacy_export(args.config, args.weights, args.output_dir, prefix=args.prefix))
