"""Boundary to the bundled, dynamically typed MineRL implementation."""

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import hashlib
import json
from dataclasses import asdict
from typing import Any, cast

import numpy as np

from minestudio.core import (
    BackendError,
    Info,
    MinecraftAction,
    MissingDependencyError,
    Observation,
    StepResult,
)
from minestudio.envs.config import EnvConfig
from minestudio.envs.engine import require_engine


class MineRLBackend:
    """Use HumanSurvival directly, bypassing the v1 simulator and its callbacks."""

    def __init__(self, config: EnvConfig) -> None:
        root = require_engine()
        jar_hash = hashlib.sha256()
        with (root / "build/libs/mcprec-6.13.jar").open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                jar_hash.update(chunk)
        self._engine_identity: Info = {"path": str(root), "jar_sha256": jar_hash.hexdigest()}
        install_manifest = root / "minestudio-install.json"
        if install_manifest.exists():
            self._engine_identity["installation"] = json.loads(install_manifest.read_text())
        try:
            import cv2

            from minestudio.simulator.minerl.herobraine.env_specs.human_survival_specs import (
                HumanSurvival,
            )
        except ImportError as error:
            raise MissingDependencyError(
                "Install minestudio[envs] for the MineRL backend"
            ) from error
        self._resize = cv2.resize
        self.config = config
        self._rng = np.random.RandomState(0)
        self._env: Any = HumanSurvival(
            rng=self._rng,
            resolution=(config.render_size.width, config.render_size.height),
        ).make(is_fault_tolerant=False)

    def _observation(self, raw: dict[str, Any]) -> Observation:
        image = np.asarray(raw["pov"], dtype=np.uint8)
        size = self.config.image_size
        image = self._resize(image, (size.width, size.height))
        return {"image": np.asarray(image, dtype=np.uint8).copy()}

    def reset(
        self, *, seed: int | None = None, options: Info | None = None
    ) -> tuple[Observation, Info]:
        if options:
            raise ValueError("MineRL reset options are not supported in this alpha")
        try:
            self._rng.seed(int(np.random.SeedSequence(seed).generate_state(1)[0]))
            self._env.seed(seed)
            raw = self._env.reset()
            info: Info = {}
            for _ in range(self.config.warmup_steps):
                raw, _, done, info = self._env.step(self._env.noop_action())
                if "error" in info or done:
                    raise BackendError(f"MineRL ended during warmup: {info}")
            return self._observation(raw), self._info(raw, info)
        except BackendError:
            raise
        except Exception as error:
            raise BackendError("MineRL reset failed") from error

    def step(self, action: MinecraftAction) -> StepResult:
        raw_action = self._env.noop_action()
        raw_action.update(action["buttons"])
        raw_action["camera"] = action["camera"].copy()
        try:
            raw, reward, done, info = self._env.step(raw_action)
            if "error" in info:
                raise BackendError(f"MineRL step failed: {info['error']}")
            truncated = bool(info.get("TimeLimit.truncated", False))
            return (
                self._observation(raw),
                float(reward),
                bool(done) and not truncated,
                truncated,
                self._info(raw, cast(Info, info)),
            )
        except BackendError:
            raise
        except Exception as error:
            raise BackendError("MineRL step failed") from error

    def _info(self, raw: dict[str, Any], info: Info) -> Info:
        # An observed empty inventory is known zero; a missing inventory is not.
        metrics: dict[str, object] = {}
        if "inventory" in raw:
            inventory: dict[str, int] = {}
            for stack in raw["inventory"].values():
                name = str(stack["type"]).removeprefix("minecraft:")
                inventory[name] = inventory.get(name, 0) + int(stack["quantity"])
            metrics["inventory"] = inventory
        metrics["stats"] = {k: v for k, v in raw.items() if k not in {"pov", "inventory"}}
        return {
            **info,
            "metrics": metrics,
            "backend": "minerl",
            "engine": self._engine_identity,
            "env_config": asdict(self.config),
            "backend_observation": {k: v for k, v in raw.items() if k != "pov"},
        }

    def execute_command(self, command: str) -> tuple[Observation, Info]:
        try:
            raw, _, done, info = self._env.execute_cmd(command)
            if done or "error" in info:
                raise BackendError(f"MineRL ended during task initialization: {info}")
            return self._observation(raw), self._info(raw, info)
        except BackendError:
            raise
        except Exception as error:
            raise BackendError(f"Minecraft initialization command failed: {command}") from error

    def close(self) -> None:
        self._env.close()
