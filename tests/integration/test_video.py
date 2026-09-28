import json
import shutil
import subprocess

import numpy as np
import pytest

from minestudio.actions import noop_action
from minestudio.core import Transition
from minestudio.rollout import EpisodeContext, EpisodeResult, VideoRecorder


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="requires FFmpeg"
)
def test_video_has_initial_and_final_frame_and_closes_encoder(tmp_path):
    context = EpisodeContext("run", 0, "attempt", 42)
    first = {"image": np.zeros((5, 7, 3), np.uint8)}
    final = {"image": np.full((5, 7, 3), 200, np.uint8)}
    with VideoRecorder(tmp_path) as recorder:
        recorder.on_episode_start(context)
        recorder.on_reset(context, first, {})
        recorder.on_transition(context, Transition(first, noop_action(), 0, final, True, False))
        recorder.on_episode_end(EpisodeResult(context, "completed", 1, 0, terminated=True))
        assert not recorder._videos
    path = recorder.paths["attempt"]
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    assert stream == {"width": 8, "height": 6, "nb_read_frames": "2"}
