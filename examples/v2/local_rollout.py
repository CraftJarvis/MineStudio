"""Run the v2 interfaces without Minecraft or Torch: python examples/v2/local_rollout.py."""

import json
from dataclasses import asdict

import numpy as np

from minestudio.actions import noop_action
from minestudio.core import Info, MinecraftAction, Observation, StepResult
from minestudio.rollout import MemoryRecorder, RolloutConfig, RolloutRunner


class DemoEnv:
    """Synthetic RGB counter environment; not a Minecraft simulator."""

    def __init__(self) -> None:
        self.steps = 0

    def reset(
        self, *, seed: int | None = None, options: Info | None = None
    ) -> tuple[Observation, Info]:
        self.steps = 0
        return {"image": np.zeros((8, 12, 3), np.uint8)}, {"seed": seed}

    def step(self, action: MinecraftAction) -> StepResult:
        self.steps += 1
        return {"image": np.full((8, 12, 3), self.steps, np.uint8)}, 1.0, self.steps == 3, False, {}

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
        return action, (0 if state is None else state) + 1


def main() -> None:
    recorder = MemoryRecorder()
    with RolloutRunner(
        DemoEnv, WalkPolicy, RolloutConfig(num_episodes=2, num_envs=2), recorders=[recorder]
    ) as runner:
        result = runner.collect()
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
