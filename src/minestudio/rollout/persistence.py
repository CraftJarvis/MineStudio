"""Connect rollout lifecycle events to incremental, checked trajectory storage."""

from dataclasses import asdict
from pathlib import Path
from types import TracebackType

from minestudio import __version__
from minestudio.core import Info, Observation, Transition
from minestudio.data.storage import TrajectoryWriter
from minestudio.rollout.records import EpisodeContext, EpisodeResult


class TrajectoryRecorder:
    """Write output_dir/run_id/attempt_id; use a context manager for abort cleanup."""

    def __init__(self, output_dir: str | Path, *, fps: float = 20.0) -> None:
        self.output_dir = Path(output_dir)
        self.fps = fps
        self._writers: dict[str, TrajectoryWriter] = {}
        self.paths: dict[str, Path] = {}

    def on_episode_start(self, context: EpisodeContext) -> None:
        path = self.output_dir / context.run_id / context.attempt_id
        writer = TrajectoryWriter(
            path,
            metadata={"context": asdict(context), "minestudio_version": __version__},
            fps=self.fps,
        )
        self._writers[context.attempt_id] = writer
        self.paths[context.attempt_id] = path

    def on_reset(self, context: EpisodeContext, observation: Observation, info: Info) -> None:
        self._writers[context.attempt_id].write_reset(observation, info)

    def on_transition(self, context: EpisodeContext, transition: Transition) -> None:
        self._writers[context.attempt_id].write_transition(transition)

    def on_episode_end(self, result: EpisodeResult) -> None:
        writer = self._writers[result.context.attempt_id]
        writer.finish(asdict(result))
        del self._writers[result.context.attempt_id]

    def close(self) -> None:
        errors: list[Exception] = []
        try:
            for writer in self._writers.values():
                try:
                    writer.close()
                except Exception as error:
                    errors.append(error)
        finally:
            self._writers.clear()
        if errors:
            raise errors[0]

    def __enter__(self) -> "TrajectoryRecorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
