"""Streaming RGB video recorder using an explicitly available FFmpeg executable."""

import shutil
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from minestudio.core import Info, MissingDependencyError, Observation, Transition
from minestudio.core.types import snapshot_observation
from minestudio.rollout.records import EpisodeContext, EpisodeResult


@dataclass
class _Video:
    process: subprocess.Popen[bytes]
    log: BinaryIO
    path: Path
    shape: tuple[int, ...]


class VideoRecorder:
    """Save T+1 policy-resolution frames per attempt, including partial failures.

    Videos are for inspection; lossless trajectory NPY files remain authoritative.
    An odd image dimension is padded by FFmpeg, not rescaled.
    """

    def __init__(self, output_dir: str | Path, *, fps: int = 20, ffmpeg: str = "ffmpeg") -> None:
        executable = shutil.which(ffmpeg)
        if executable is None:
            raise MissingDependencyError("VideoRecorder requires ffmpeg on PATH")
        if type(fps) is not int or fps < 1:
            raise ValueError("fps must be a positive integer")
        self.output_dir = Path(output_dir)
        self.fps = fps
        self._ffmpeg = executable
        self._videos: dict[str, _Video] = {}
        self.paths: dict[str, Path] = {}

    def on_episode_start(self, context: EpisodeContext) -> None:
        pass

    def on_reset(self, context: EpisodeContext, observation: Observation, info: Info) -> None:
        image = snapshot_observation(observation)["image"]
        path = self.output_dir / context.run_id / "videos" / f"{context.attempt_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.with_suffix(".partial.mp4").exists():
            raise FileExistsError(path)
        temporary = path.with_suffix(".partial.mp4")
        log = path.with_suffix(".ffmpeg.log").open("xb")
        try:
            process = subprocess.Popen(
                [
                    self._ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{image.shape[1]}x{image.shape[0]}",
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=log,
            )
        except BaseException:
            log.close()
            raise
        self._videos[context.attempt_id] = _Video(process, log, temporary, image.shape)
        self.paths[context.attempt_id] = path
        self._write(context, observation)

    def _write(self, context: EpisodeContext, observation: Observation) -> None:
        video = self._videos[context.attempt_id]
        image = snapshot_observation(observation)["image"]
        if image.shape != video.shape:
            raise ValueError("Video frame size changed during an episode")
        if video.process.stdin is None:
            raise RuntimeError("Video encoder has no input stream")
        try:
            video.process.stdin.write(image.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(
                f"FFmpeg encoding failed; inspect {self.paths[context.attempt_id].with_suffix('.ffmpeg.log')}"
            ) from error

    def on_transition(self, context: EpisodeContext, transition: Transition) -> None:
        self._write(context, transition.next_observation)

    def on_episode_end(self, result: EpisodeResult) -> None:
        self._finish(result.context.attempt_id)

    def _finish(self, attempt_id: str) -> None:
        video = self._videos.pop(attempt_id, None)
        if video is None:
            return
        try:
            if video.process.stdin is not None:
                with suppress(BrokenPipeError):
                    video.process.stdin.close()
            try:
                returncode = video.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                video.process.kill()
                video.process.wait(timeout=5)
                raise RuntimeError("FFmpeg did not finish within 15 seconds") from None
            if returncode:
                raise RuntimeError(
                    f"FFmpeg exited with code {returncode}; see {self.paths[attempt_id].with_suffix('.ffmpeg.log')}"
                )
            video.path.rename(self.paths[attempt_id])
        finally:
            if video.process.poll() is None:
                video.process.kill()
                video.process.wait(timeout=5)
            video.log.close()

    def close(self) -> None:
        errors: list[Exception] = []
        for attempt_id in list(self._videos):
            try:
                self._finish(attempt_id)
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
