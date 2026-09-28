"""Optional Torch convenience base; heads and training algorithms are not required."""

try:
    import torch
    from torch import nn
except ImportError as error:
    from minestudio.core import MissingDependencyError

    raise MissingDependencyError("Install minestudio[policies] for Torch policies") from error


class TorchPolicy(nn.Module):
    """A policy's parameters own their device; recurrent state belongs to callers."""

    @property
    def device(self) -> torch.device:
        parameter = next(self.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")
