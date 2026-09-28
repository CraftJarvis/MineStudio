import json

import numpy as np
import pytest

from minestudio.actions import noop_action
from minestudio.core import Transition
from minestudio.data import TrajectoryDataset, TrajectorySubset, TrajectoryWriter


def record(path):
    frames = [{"image": np.full((4, 4, 3), i, np.uint8)} for i in range(6)]
    with TrajectoryWriter(path, metadata={}) as writer:
        writer.write_reset(frames[0], {})
        for i in range(5):
            action = noop_action()
            action["buttons"]["attack"] = i % 2
            writer.write_transition(
                Transition(frames[i], action, float(i), frames[i + 1], False, i == 4)
            )
        writer.finish({"status": "completed", "num_steps": 5})


def test_native_windows_and_padding(tmp_path):
    path = tmp_path / "episode"
    record(path)
    with TrajectoryDataset(path, sequence_length=3) as dataset:
        assert len(dataset) == 2
        a, b = dataset[0], dataset[1]
        assert a.observations["image"][:, 0, 0, 0].tolist() == [0, 1, 2, 3]
        assert b.observations["image"][:, 0, 0, 0].tolist() == [3, 4, 5, 0]
        assert b.valid_mask.tolist() == [True, True, False]
        assert b.observation_mask["image"].tolist() == [True, True, True, False]
        assert a.first_mask.tolist() == [True, False, False]
        assert not b.first_mask.any()
        assert b.truncated.tolist() == [False, True, False]
        assert b.reward_mask.tolist() == b.boundary_mask.tolist() == [True, True, False]
        subset = TrajectorySubset(dataset, [1])
        assert subset[0].start_step == 3
    with pytest.raises(RuntimeError, match="closed"):
        dataset[0]
    with TrajectoryDataset(path, sequence_length=3, pad_end=False) as dataset:
        assert len(dataset) == 1


def test_manifest_rejects_split_leakage(tmp_path):
    path = tmp_path / "data.json"
    data = {
        "format": "minestudio.dataset",
        "schema_version": 1,
        "records": [
            {
                "episode_id": "a",
                "source_group": "same",
                "split": "train",
                "frames": 5,
                "storage": "trajectory",
            },
            {
                "episode_id": "b",
                "source_group": "same",
                "split": "test",
                "frames": 5,
                "storage": "trajectory",
            },
        ],
    }
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="multiple splits"):
        TrajectoryDataset(path, sequence_length=3, split="train")
