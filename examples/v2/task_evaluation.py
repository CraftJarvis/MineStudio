"""Exercise tasks, evaluation, trajectory readback and optional video without Minecraft."""

import argparse
from pathlib import Path

import numpy as np

from minestudio.actions import noop_action
from minestudio.core import ImageSize, Info, MinecraftAction, Observation, StepResult
from minestudio.data.storage import TrajectoryReader
from minestudio.envs import EnvConfig, MinecraftEnv
from minestudio.evaluation import EvaluationConfig, EvaluationSuite, evaluate
from minestudio.tasks import EndRule, InitializationRule, RewardRule, TaskSpec


class CounterBackend:
    """Synthetic task fixture. Quantity increases when forward is pressed."""

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.quantity = 0

    def observation(self) -> Observation:
        size = self.config.image_size
        image = np.zeros((size.height, size.width, 3), np.uint8)
        image[:, : min(size.width, self.quantity * 20), 1] = 180
        return {"image": image}

    def info(self) -> Info:
        return {"metrics": {"quantity": self.quantity}, "backend": "synthetic-counter"}

    def reset(
        self, *, seed: int | None = None, options: Info | None = None
    ) -> tuple[Observation, Info]:
        self.quantity = 0
        return self.observation(), self.info()

    def execute_command(self, command: str) -> tuple[Observation, Info]:
        if command != "/clear":
            raise ValueError("Demo supports only /clear")
        self.quantity = 0
        return self.observation(), self.info()

    def step(self, action: MinecraftAction) -> StepResult:
        self.quantity += action["buttons"]["forward"]
        return self.observation(), 0.0, False, False, self.info()

    def close(self) -> None:
        pass


class WalkPolicy:
    def initial_state(self) -> int:
        return 0

    def act(
        self, observation: Observation, state: int | None = None, *, deterministic: bool = False
    ) -> tuple[MinecraftAction, int]:
        action = noop_action()
        action["buttons"]["forward"] = 1
        return action, (state or 0) + 1


def make_env(*, task: TaskSpec) -> MinecraftEnv:
    return MinecraftEnv(
        EnvConfig(image_size=ImageSize(96, 160)), task=task, backend_factory=CounterBackend
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/task-demo"))
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    task = TaskSpec(
        "demo_collect",
        "1",
        "Collect three demo items",
        initialization=(InitializationRule("/clear"),),
        rewards=(RewardRule("items", ("metrics", "quantity")),),
        success=(EndRule(("metrics", "quantity"), 3),),
        max_episode_steps=5,
    )
    result = evaluate(
        WalkPolicy,
        suite=EvaluationSuite("demo", "1", (task,), (42, 43)),
        env_factory=make_env,
        config=EvaluationConfig(
            "walk",
            "example-v1",
            "rgb-v1",
            "minecraft-native-v1",
            output_dir=args.output_dir,
            record_video=args.video,
            num_envs=2,
        ),
    )
    for case in result.cases:
        path = args.output_dir / result.run_id / case.context.attempt_id
        episode = TrajectoryReader(path).read().as_episode()
        assert len(episode.observations) == 4
    print(
        f"Synthetic demo: {result.completed}/{result.requested} completed; success rate {result.success_rate}"
    )
    print(f"Verified T+1 observation readback. Report: {result.report_path}")


if __name__ == "__main__":
    main()
