"""Errors shared by MineStudio domains."""


class MineStudioError(Exception):
    """Base class for errors raised by MineStudio."""


class MissingDependencyError(MineStudioError, ImportError):
    """An explicitly requested feature needs an optional dependency."""


class EngineNotFoundError(MineStudioError, FileNotFoundError):
    """The Minecraft engine has not been installed."""


class BackendError(MineStudioError, RuntimeError):
    """An environment operation failed without a valid transition."""


class EpisodeStateError(MineStudioError, RuntimeError):
    """An operation is invalid in the current episode lifecycle."""
