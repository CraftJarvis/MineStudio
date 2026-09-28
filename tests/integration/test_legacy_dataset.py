# ruff: noqa: E402
"""Legacy storage boundaries, absent labels and reference integrity."""

import io
import json
import pickle

import numpy as np
import pytest

av = pytest.importorskip("av")
lmdb = pytest.importorskip("lmdb")

from minestudio.actions import noop_action
from minestudio.data import TrajectoryDataset
from minestudio.data.legacy_lmdb import _ArrayUnpickler


def make_dataset(root):
    modalities = {}
    for modality in ("image", "action"):
        path = root / modality
        path.mkdir()
        env = lmdb.open(str(path), map_size=10 * 1024**2)
        info = {"episode": "record", "episode_idx": 0, "num_frames": 8}
        with env.begin(write=True) as txn:
            txn.put(b"__chunk_size__", pickle.dumps(4))
            txn.put(b"__chunk_infos__", pickle.dumps([info]))
            for start in (0, 4):
                if modality == "image":
                    stream = io.BytesIO()
                    with av.open(stream, "w", format="mp4") as video:
                        encoder = video.add_stream("libx264", rate=20)
                        encoder.width, encoder.height = 16, 16
                        encoder.pix_fmt = "yuv420p"
                        for i in range(start, start + 4):
                            frame = av.VideoFrame.from_ndarray(
                                np.full((16, 16, 3), i * 25, np.uint8), format="rgb24"
                            )
                            for packet in encoder.encode(frame):
                                video.mux(packet)
                        for packet in encoder.encode():
                            video.mux(packet)
                    payload = stream.getvalue()
                else:
                    data = {k: np.zeros(4, np.int64) for k in noop_action()["buttons"]}
                    data["attack"][:] = np.arange(start, start + 4) % 2
                    data["camera"] = np.column_stack([np.arange(start, start + 4), np.zeros(4)])
                    payload = pickle.dumps(data)
                txn.put(str((0, start)).encode(), payload)
        env.close()
        modalities[modality] = {**info, "chunk_size": 4, "path": f"{modality}/data.mdb"}
    manifest = {
        "format": "minestudio.dataset",
        "schema_version": 1,
        "records": [
            {
                "episode_id": "record",
                "source_group": "source",
                "split": "train",
                "storage": "legacy_lmdb",
                "frames": 8,
                "modalities": modalities,
            },
        ],
    }
    path = root / "dataset.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_legacy_cross_chunk_alignment_tail_and_unknowns(tmp_path):
    path, manifest = make_dataset(tmp_path)
    with TrajectoryDataset(path, sequence_length=5, stride=3) as dataset:
        middle = dataset[1]
        assert middle.start_step == 3
        assert [a["camera"][0] for a in middle.actions] == [3, 4, 5, 6, 7]
        assert [a["buttons"]["attack"] for a in middle.actions] == [1, 0, 1, 0, 1]
        np.testing.assert_allclose(
            middle.observations["image"][:5].mean(axis=(1, 2, 3)), np.arange(3, 8) * 25, atol=2
        )
        assert middle.valid_mask.all()
        assert middle.observation_mask["image"].tolist() == [True] * 5 + [False]
        tail = dataset[-1]
        assert tail.valid_mask.tolist() == [True, True, False, False, False]
        assert tail.observation_mask["image"].tolist() == [True, True, False, False, False, False]
        for window in (middle, tail):
            assert window.timebase == "legacy_rows"
            assert not window.reward_mask.any()
            assert not window.boundary_mask.any()
            assert not window.first_mask.any()
    manifest["records"][0]["modalities"]["image"]["episode_idx"] = 1
    path.write_text(json.dumps(manifest))
    with (
        TrajectoryDataset(path, sequence_length=3) as dataset,
        pytest.raises(ValueError, match="identity/length"),
    ):
        dataset[0]


def test_legacy_pickle_rejects_executable_globals():
    with pytest.raises(ValueError, match="Unsupported legacy pickle global"):
        _ArrayUnpickler(io.BytesIO(pickle.dumps(eval))).load()
