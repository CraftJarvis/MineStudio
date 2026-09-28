import numpy as np
import pytest

from minestudio.actions import noop_action
from minestudio.core import Transition
from minestudio.core.serialization import decode_value, encode_value
from minestudio.data.storage import TrajectoryReader, TrajectoryWriter


def observation(value):
    return {"image": np.full((4, 6, 3), value, np.uint8)}


def transition(first=0, final=1, *, terminated=True):
    action = noop_action()
    action["camera"][:] = [1.25, -2.5]
    return Transition(
        observation(first),
        action,
        3.25,
        observation(final),
        terminated,
        False,
        {"array": np.arange(6, dtype=np.int16).reshape(2, 3), "nested": {4: ("item", 2)}},
    )


def test_lossless_roundtrip_and_t_plus_one(tmp_path):
    directory = tmp_path / "attempt"
    first = transition(terminated=False)
    last = transition(1, 2)
    with TrajectoryWriter(directory, metadata={"seed": 3, "task": "collect"}) as writer:
        writer.write_reset(observation(0), {"time": np.array(42, dtype=np.int64)})
        writer.write_transition(first)
        writer.write_transition(last)
        writer.finish({"status": "completed", "num_steps": 2, "success": True})
    recorded = TrajectoryReader(directory).read()
    episode = recorded.as_episode()
    assert recorded.metadata == {"seed": 3, "task": "collect"}
    assert len(episode.observations) == 3
    assert [int(o["image"].max()) for o in episode.observations] == [0, 1, 2]
    loaded = episode.transitions[0]
    np.testing.assert_array_equal(loaded.action["camera"], first.action["camera"])
    assert loaded.info["array"].dtype == np.int16
    np.testing.assert_array_equal(loaded.info["array"], first.info["array"])
    assert loaded.info["nested"] == {4: ("item", 2)}
    assert recorded.initial_info["time"].shape == ()
    assert loaded.reward == 3.25
    with pytest.raises(FileExistsError):
        TrajectoryWriter(directory, metadata={})


def test_failed_attempt_has_no_synthetic_boundary(tmp_path):
    with TrajectoryWriter(tmp_path / "attempt", metadata={}) as writer:
        writer.write_reset(observation(0), {})
        writer.write_transition(transition(terminated=False))
        writer.finish({"status": "failed", "num_steps": 1})
    recorded = TrajectoryReader(tmp_path / "attempt").read()
    assert len(recorded.transitions) == 1
    assert not recorded.transitions[0].terminated
    with pytest.raises(ValueError, match="not a completed"):
        recorded.as_episode()


def test_interrupted_recording_requires_explicit_recovery(tmp_path):
    with TrajectoryWriter(tmp_path / "attempt", metadata={}) as writer:
        writer.write_reset(observation(0), {})
        writer.write_transition(transition(terminated=False))
    reader = TrajectoryReader(tmp_path / "attempt")
    with pytest.raises(ValueError, match="recover_incomplete"):
        reader.read()
    assert len(reader.read(recover_incomplete=True).transitions) == 1


def test_crash_recovers_only_committed_prefix(tmp_path):
    directory = tmp_path / "attempt"
    writer = TrajectoryWriter(directory, metadata={})
    writer.write_reset(observation(0), {})
    writer.write_transition(transition(terminated=False))
    # Simulate process death after a partial JSONL append: no close/final manifest.
    writer._journal.write('{"index": 1')
    writer._journal.close()
    np.save(directory / "observations" / "00000002.npy", observation(2)["image"])
    result = TrajectoryReader(directory).read(recover_incomplete=True)
    assert result.status == "recording" and len(result.transitions) == 1
    assert result.outcome is None


@pytest.mark.parametrize("target", ["observations/00000001.npy", "transitions.jsonl", "reset.json"])
def test_corruption_is_detected(tmp_path, target):
    directory = tmp_path / "attempt"
    with TrajectoryWriter(directory, metadata={}) as writer:
        writer.write_reset(observation(0), {})
        writer.write_transition(transition())
        writer.finish({"status": "completed", "num_steps": 1})
    with (directory / target).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        TrajectoryReader(directory).read()


def test_info_encoding_does_not_confuse_user_keys_with_tags():
    value = {"type": "ndarray", "shape": [2], "nested": {1: np.array([True, False])}}
    result = decode_value(encode_value(value))
    assert result["type"] == "ndarray"
    np.testing.assert_array_equal(result["nested"][1], value["nested"][1])
    with pytest.raises(TypeError):
        encode_value(np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        decode_value({"type": "ndarray", "dtype": "O", "shape": [1], "data": "AAAA"})
