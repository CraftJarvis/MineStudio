"""Incremental native trajectory storage with checksums and explicit recovery.

Images are uint8 NPY files, and transitions are an append-only JSONL journal.
Only a successfully written journal line commits its next observation. Loading
never executes pickle or imports a model/environment runtime.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import numpy as np

from minestudio.actions import copy_action
from minestudio.core import Episode, Info, Observation, Transition
from minestudio.core.serialization import decode_value, encode_value
from minestudio.core.types import snapshot_observation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Replace a JSON document atomically after fully serializing it."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _info(value: Any) -> Info:
    decoded = decode_value(value)
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in cast(dict[object, object], decoded)
    ):
        raise ValueError("Top-level info must be a string-keyed mapping")
    return cast(Info, decoded)


class TrajectoryWriter:
    """Own one attempt directory; keep only the latest image in memory."""

    def __init__(self, directory: str | Path, *, metadata: Info, fps: float = 20.0) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        self.directory = Path(directory)
        # Validate metadata before creating a partial artifact.
        encoded = encode_value(metadata)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "observations").mkdir()
        self._manifest: dict[str, Any] = {
            "format": "minestudio.trajectory",
            "schema_version": 1,
            "status": "recording",
            "metadata": encoded,
            "fps": fps,
            "action_format": "minecraft-native-v1",
            "observation_format": "rgb-uint8-hwc-v1",
            "num_transitions": 0,
        }
        self._previous: Observation | None = None
        self._count = 0
        self._boundary = False
        self._closed = False
        write_json(self.directory / "manifest.json", self._manifest)
        self._journal = (self.directory / "transitions.jsonl").open("x", encoding="utf-8")

    def _frame(self, index: int, observation: Observation) -> dict[str, object]:
        image = snapshot_observation(observation)["image"]
        path = self.directory / "observations" / f"{index:08d}.npy"
        with path.open("xb") as stream:
            np.save(stream, image, allow_pickle=False)
        return {"path": str(path.relative_to(self.directory)), "sha256": sha256_file(path)}

    def write_reset(self, observation: Observation, info: Info) -> None:
        if self._closed or self._previous is not None:
            raise RuntimeError("Writer is closed or already has its initial observation")
        packed = encode_value(info)
        frame = self._frame(0, observation)
        write_json(self.directory / "reset.json", {"observation": frame, "info": packed})
        self._previous = snapshot_observation(observation)

    def write_transition(self, transition: Transition) -> None:
        if self._closed or self._previous is None or self._boundary:
            raise RuntimeError("Cannot append before reset, after a boundary, or after close")
        if not np.array_equal(self._previous["image"], transition.observation["image"]):
            raise ValueError("Transition is not temporally aligned")
        next_observation = snapshot_observation(transition.next_observation)
        if next_observation["image"].shape != self._previous["image"].shape:
            raise ValueError("Image shape changed within an episode")
        action = copy_action(transition.action)
        if (
            not math.isfinite(transition.reward)
            or type(transition.terminated) is not bool
            or type(transition.truncated) is not bool
        ):
            raise ValueError("Invalid reward or boundary flag")
        record: dict[str, Any] = {
            "index": self._count,
            "action": {"buttons": action["buttons"], "camera": action["camera"].tolist()},
            "reward": transition.reward,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "info": encode_value(transition.info),
        }
        record["next_observation"] = self._frame(self._count + 1, next_observation)
        self._journal.write(json.dumps(record, allow_nan=False) + "\n")
        self._journal.flush()
        self._previous = next_observation
        self._count += 1
        self._boundary = transition.terminated or transition.truncated

    def finish(self, result: Info) -> None:
        if self._closed:
            raise RuntimeError("Writer is already closed")
        if (
            result.get("status") not in ("completed", "failed", "cancelled")
            or result.get("num_steps") != self._count
        ):
            raise ValueError("Attempt outcome does not match the recorded transitions")
        if result["status"] == "completed" and not self._boundary:
            raise ValueError("Completed trajectory requires a real terminal/truncated transition")
        self._manifest["outcome"] = encode_value(result)
        self._finalize(str(result["status"]))

    def _finalize(self, status: str) -> None:
        self._journal.close()
        self._manifest.update(
            status=status,
            num_transitions=self._count,
            journal_sha256=sha256_file(self.directory / "transitions.jsonl"),
        )
        reset = self.directory / "reset.json"
        if reset.exists():
            self._manifest["reset_sha256"] = sha256_file(reset)
        write_json(self.directory / "manifest.json", self._manifest)
        self._closed = True

    def close(self) -> None:
        if not self._closed:
            self._finalize("interrupted")

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True)
class RecordedTrajectory:
    metadata: Info
    outcome: Info | None
    initial_observation: Observation | None
    initial_info: Info | None
    transitions: tuple[Transition, ...]
    status: str

    def as_episode(self) -> Episode:
        if self.status != "completed" or self.initial_observation is None:
            raise ValueError("A failed/interrupted attempt is not a completed Episode")
        return Episode(self.initial_observation, self.transitions)


class TrajectoryReader:
    """Verify and read one attempt; recovery of interrupted attempts is opt-in."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()

    def _frame(self, reference: dict[str, Any]) -> Observation:
        path = (self.directory / reference["path"]).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Observation path escapes trajectory directory")
        if sha256_file(path) != reference["sha256"]:
            raise ValueError("Observation SHA-256 mismatch")
        image = np.load(path, allow_pickle=False)
        return snapshot_observation({"image": image})

    def read(self, *, recover_incomplete: bool = False) -> RecordedTrajectory:
        manifest: dict[str, Any] = json.loads((self.directory / "manifest.json").read_text())
        for key, expected in {
            "format": "minestudio.trajectory",
            "schema_version": 1,
            "action_format": "minecraft-native-v1",
            "observation_format": "rgb-uint8-hwc-v1",
        }.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Unsupported trajectory {key}")
        status = manifest["status"]
        if status not in {"recording", "interrupted", "completed", "failed", "cancelled"}:
            raise ValueError("Unknown trajectory status")
        incomplete = status in {"recording", "interrupted"}
        if incomplete and not recover_incomplete:
            raise ValueError(
                "Interrupted trajectory; use recover_incomplete=True to inspect its valid prefix"
            )
        journal = self.directory / "transitions.jsonl"
        if "journal_sha256" in manifest and sha256_file(journal) != manifest["journal_sha256"]:
            raise ValueError("Transition journal SHA-256 mismatch")
        initial: Observation | None = None
        initial_info: Info | None = None
        reset_path = self.directory / "reset.json"
        if "reset_sha256" in manifest and not reset_path.exists():
            raise ValueError("Recorded reset file is missing")
        if reset_path.exists():
            if "reset_sha256" in manifest and sha256_file(reset_path) != manifest["reset_sha256"]:
                raise ValueError("Reset SHA-256 mismatch")
            reset = json.loads(reset_path.read_text())
            initial, initial_info = self._frame(reset["observation"]), _info(reset["info"])
        transitions: list[Transition] = []
        previous = initial
        with journal.open(encoding="utf-8") as stream:
            for line in stream:
                if incomplete and not line.endswith("\n"):
                    break
                item = json.loads(line)
                if previous is None or item["index"] != len(transitions):
                    raise ValueError("Missing initial observation or out-of-order transition")
                if transitions and (transitions[-1].terminated or transitions[-1].truncated):
                    raise ValueError("Transition found after an episode boundary")
                observation = self._frame(item["next_observation"])
                action = copy_action(
                    {
                        "buttons": item["action"]["buttons"],
                        "camera": np.asarray(item["action"]["camera"], dtype=np.float32),
                    }
                )
                if (
                    type(item["terminated"]) is not bool
                    or type(item["truncated"]) is not bool
                    or not math.isfinite(item["reward"])
                ):
                    raise ValueError("Malformed reward/boundary")
                transitions.append(
                    Transition(
                        previous,
                        action,
                        float(item["reward"]),
                        observation,
                        item["terminated"],
                        item["truncated"],
                        _info(item["info"]),
                    )
                )
                previous = observation
        if not incomplete and len(transitions) != manifest["num_transitions"]:
            raise ValueError("Manifest transition count mismatch")
        outcome = _info(manifest["outcome"]) if "outcome" in manifest else None
        if not incomplete and (
            outcome is None
            or outcome.get("status") != status
            or outcome.get("num_steps") != len(transitions)
        ):
            raise ValueError("Manifest outcome mismatch")
        result = RecordedTrajectory(
            _info(manifest["metadata"]), outcome, initial, initial_info, tuple(transitions), status
        )
        if status == "completed":
            result.as_episode()
        return result
