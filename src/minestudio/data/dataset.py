"""Framework-independent fixed-split windows over native and legacy trajectories."""

import hashlib
import json
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from minestudio.actions import noop_action, validate_action
from minestudio.core import TrajectoryWindow
from minestudio.data.storage import TrajectoryReader


class TrajectoryDataset:
    """Read a minestudio.dataset v1 manifest or one completed native trajectory.

    Splits are assigned to records/source groups before windowing. No sampling,
    frame skipping, implicit split generation or terminal inference occurs here.
    Native records are cached one at a time; legacy windows decode only needed chunks.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        sequence_length: int,
        stride: int | None = None,
        split: Literal["train", "validation", "test"] | None = None,
        pad_end: bool = True,
    ) -> None:
        if (
            type(sequence_length) is not int
            or sequence_length <= 0
            or (stride is not None and (type(stride) is not int or stride <= 0))
        ):
            raise ValueError("sequence_length and stride must be positive")
        path = Path(path).resolve()
        if path.is_dir():
            source = json.loads((path / "manifest.json").read_text())
            manifest = {
                "format": "minestudio.dataset",
                "schema_version": 1,
                "records": [
                    {
                        "episode_id": path.name,
                        "source_group": path.name,
                        "storage": "trajectory",
                        "path": str(path),
                        "frames": source["num_transitions"],
                        "identity": source,
                    }
                ],
            }
        else:
            manifest = json.loads(path.read_text())
        if manifest.get("format") != "minestudio.dataset" or manifest.get("schema_version") != 1:
            raise ValueError("Expected minestudio.dataset schema_version=1")
        self.root = (path.parent / manifest.get("dataset_root", ".")).resolve()
        self.sequence_length = sequence_length
        self.stride = sequence_length if stride is None else stride
        self.pad_end = pad_end
        self.split = split
        groups, identities = {}, set()
        for record in manifest["records"]:
            identity = record["episode_id"]
            if (
                not isinstance(identity, str)
                or not identity
                or identity in identities
                or (type(record["frames"]) is not int or record["frames"] <= 0)
            ):
                raise ValueError("Records need unique identities and positive lengths")
            identities.add(identity)
            group = record["source_group"]
            assignment = record.get("split")
            if (
                not isinstance(group, str)
                or not group
                or assignment not in (None, "train", "validation", "test")
            ):
                raise ValueError("Invalid source group or split")
            if group in groups and groups[group] != assignment:
                raise ValueError("A source group occurs in multiple splits")
            groups[group] = assignment
            if record["storage"] not in ("trajectory", "legacy_lmdb"):
                raise ValueError("Unsupported dataset storage")
            if record["storage"] == "legacy_lmdb":
                for modality in ("image", "action"):
                    ref = record["modalities"][modality]
                    if ref["num_frames"] != record["frames"] or ref["episode"] != identity:
                        raise ValueError("Legacy modality identity/length mismatch")
        self.records = [r for r in manifest["records"] if split is None or r.get("split") == split]
        if not self.records:
            raise ValueError("No records in the requested split")
        self._ends = [0]
        for record in self.records:
            available = record["frames"] if pad_end else record["frames"] - sequence_length + 1
            self._ends.append(self._ends[-1] + max(0, (available + self.stride - 1) // self.stride))
        identity = {
            "manifest": manifest,
            "length": sequence_length,
            "stride": self.stride,
            "split": split,
            "pad_end": pad_end,
        }
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self._cached_record = None
        self._cached_trajectory = None
        self._closed = False

    def __len__(self) -> int:
        return self._ends[-1]

    def location(self, index: int) -> tuple[int, int]:
        """Stable record index and start; useful for explicit sampling schedules."""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        record = bisect_right(self._ends, index) - 1
        return record, (index - self._ends[record]) * self.stride

    def __getitem__(self, index: int) -> TrajectoryWindow:
        if self._closed:
            raise RuntimeError("Dataset is closed")
        record_index, start = self.location(index)
        record = self.records[record_index]
        length = self.sequence_length
        count = min(length, record["frames"] - start)

        def zeros():
            return np.zeros(length, dtype=np.bool_)

        valid, action_mask, reward_mask, boundary_mask, first = (zeros() for _ in range(5))
        terminated, truncated = zeros(), zeros()
        rewards = np.zeros(length, dtype=np.float32)
        actions = [noop_action() for _ in range(length)]
        valid[:count] = action_mask[:count] = True
        if record["storage"] == "legacy_lmdb":
            from minestudio.data.legacy_lmdb import read_chunks

            refs = record["modalities"]
            frames = read_chunks(
                self.root,
                refs["image"],
                start,
                min(start + count + 1, record["frames"]),
                images=True,
            )
            raw = read_chunks(self.root, refs["action"], start, start + count, images=False)
            if set(raw) != set(actions[0]["buttons"]) | {"camera"}:
                raise ValueError("Unexpected legacy action keys")
            if raw["camera"].shape != (count, 2) or not np.isfinite(raw["camera"]).all():
                raise ValueError("Invalid legacy camera values")
            for key, values in raw.items():
                if key != "camera" and (
                    values.shape != (count,) or not np.isin(values, [0, 1]).all()
                ):
                    raise ValueError("Invalid legacy buttons")
            for i in range(count):
                actions[i] = {
                    "buttons": {k: int(v[i]) for k, v in raw.items() if k != "camera"},
                    "camera": raw["camera"][i].astype(np.float32),
                }
        else:
            if self._cached_record != record_index:
                trajectory = TrajectoryReader(self.root / record["path"]).read()
                episode = trajectory.as_episode()
                if len(episode.transitions) != record["frames"]:
                    raise ValueError("Native record length disagrees with manifest")
                self._cached_trajectory = episode
                self._cached_record = record_index
            episode = self._cached_trajectory
            selected = episode.transitions[start : start + count]
            frames = np.stack(
                [t.observation["image"] for t in selected]
                + [selected[-1].next_observation["image"]]
            )
            for i, transition in enumerate(selected):
                actions[i] = {
                    "buttons": dict(transition.action["buttons"]),
                    "camera": transition.action["camera"].copy(),
                }
                rewards[i] = transition.reward
                terminated[i], truncated[i] = transition.terminated, transition.truncated
            reward_mask[:count] = boundary_mask[:count] = True
            first[0] = start == 0
        for action in actions:
            validate_action(action)
        images = np.zeros((length + 1, *frames.shape[1:]), dtype=np.uint8)
        images[: len(frames)] = frames
        image_mask = np.arange(length + 1) < len(frames)
        return TrajectoryWindow(
            record["episode_id"],
            start,
            {"image": images},
            tuple(actions),
            rewards,
            terminated,
            truncated,
            valid,
            {"image": image_mask},
            action_mask,
            reward_mask,
            boundary_mask,
            first,
            timebase="legacy_rows" if record["storage"] == "legacy_lmdb" else "environment_steps",
        )

    def close(self) -> None:
        self._cached_trajectory = None
        self._closed = True

    def __enter__(self) -> "TrajectoryDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class TrajectorySubset:
    """Explicit window selection after the parent split; never closes its parent."""

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]) -> None:
        if not indices or any(type(i) is not int or not 0 <= i < len(dataset) for i in indices):
            raise ValueError("Subset requires nonempty in-range indices")
        self.dataset = dataset
        self.indices = tuple(indices)
        self.fingerprint = hashlib.sha256(
            json.dumps([dataset.fingerprint, self.indices]).encode()
        ).hexdigest()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> TrajectoryWindow:
        return self.dataset[self.indices[index]]
