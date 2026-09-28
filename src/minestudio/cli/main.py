"""Read-only diagnostics and explicit engine installation."""

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

from minestudio import __version__
from minestudio.envs.engine import engine_root, install_engine


def main() -> int:
    parser = argparse.ArgumentParser(prog="minestudio")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect dependencies and engine without launching anything")
    engine = commands.add_parser("engine", help="Manage the Minecraft engine")
    operations = engine.add_subparsers(dest="operation", required=True)
    install = operations.add_parser("install", help="Verify and install a local engine.zip")
    install.add_argument("archive", type=Path)
    install.add_argument("--sha256", required=True)
    validate = commands.add_parser("task", help="Validate and print a task configuration")
    validate.add_argument("source", help="Built-in task id or JSON/YAML path")
    trajectory = commands.add_parser("trajectory", help="Inspect and verify a native trajectory")
    trajectory.add_argument("directory", type=Path)
    trajectory.add_argument("--recover", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Run a VPT export on a Minecraft task")
    evaluate.add_argument("--task", required=True)
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--suite-id", default="local-evaluation")
    evaluate.add_argument("--seeds", type=int, nargs="+", default=[42])
    evaluate.add_argument("--repeats", type=int, default=1)
    evaluate.add_argument("--num-envs", type=int, default=1)
    evaluate.add_argument("--max-retries", type=int, default=0)
    evaluate.add_argument("--max-episode-steps", type=int)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--video", action="store_true")
    convert = commands.add_parser(
        "convert-vpt", help="Convert a tensor-only v1 state dict and JSON config"
    )
    convert.add_argument("--config", type=Path, required=True)
    convert.add_argument("--weights", type=Path, required=True)
    convert.add_argument("--output-dir", type=Path, required=True)
    convert.add_argument("--prefix", default="")
    args = parser.parse_args()
    if args.command == "doctor":
        print(
            json.dumps(
                {
                    "version": __version__,
                    "engine_path": str(engine_root()),
                    "engine_present": (engine_root() / "build/libs/mcprec-6.13.jar").is_file(),
                    "java": shutil.which("java"),
                    "display": os.environ.get("DISPLAY"),
                    "xvfb": shutil.which("Xvfb"),
                    "ffmpeg": shutil.which("ffmpeg"),
                    "packages": {
                        name: importlib.util.find_spec(name) is not None
                        for name in (
                            "numpy",
                            "gymnasium",
                            "torch",
                            "cv2",
                            "ray",
                            "lightning",
                            "mkdocs",
                            "yaml",
                        )
                    },
                },
                indent=2,
            )
        )
    elif args.command == "task":
        from minestudio.cli.experiments import read_task

        print(json.dumps(read_task(args.source).to_dict(), indent=2, ensure_ascii=False))
    elif args.command == "trajectory":
        from minestudio.data.storage import TrajectoryReader

        result = TrajectoryReader(args.directory).read(recover_incomplete=args.recover)
        print(
            json.dumps(
                {
                    "status": result.status,
                    "num_steps": len(result.transitions),
                    "num_observations": len(result.transitions)
                    + int(result.initial_observation is not None),
                },
                indent=2,
            )
        )
    elif args.command == "evaluate":
        from minestudio.cli.experiments import run_evaluation

        run_evaluation(args)
    elif args.command == "convert-vpt":
        from minestudio.cli.experiments import convert_vpt

        convert_vpt(args)
    else:
        try:
            print(install_engine(args.archive, sha256=args.sha256))
        except (OSError, ValueError) as error:
            parser.exit(1, f"Engine installation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
