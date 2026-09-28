"""VPT inference with explicit state, native actions and verified local exports."""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from minestudio import __version__
from minestudio.actions import VPTAction, VPTActionCodec
from minestudio.core import ImageSize, MinecraftAction, MissingDependencyError, Observation
from minestudio.core.types import snapshot_observation
from minestudio.policies.torch_policy import TorchPolicy
from minestudio.policies.vpt.config import VPTConfig

try:
    import cv2
    import gymnasium as gym
    import torch
    from safetensors.torch import load_file, save_file

    from minestudio.policies.vpt._network import MinecraftPolicy
    from minestudio.utils.vpt_lib.action_head import make_action_head
    from minestudio.utils.vpt_lib.scaled_mse_head import ScaledMSEHead
except ImportError as error:
    raise MissingDependencyError("Install minestudio[policies] for VPTPolicy") from error


@dataclass(frozen=True)
class VPTState:
    """Private tensor tree plus the explicit first-observation flag for one stream."""

    recurrent: Any
    first: bool = True


@dataclass(frozen=True)
class VPTStep:
    """Sampled encoded action is retained; log_prob is for the sampling distribution.

    With deterministic=True, behavior is a point mass and log_prob is zero.
    value is denormalized. This is inference output, not a differentiable batch.
    """

    action: MinecraftAction
    policy_action: VPTAction
    log_prob: float
    value: float


def _state_to_device(tree: Any, device: torch.device) -> Any:
    if isinstance(tree, torch.Tensor):
        return tree.detach().clone().to(device)
    if isinstance(tree, list):
        return [_state_to_device(item, device) for item in tree]
    if isinstance(tree, tuple):
        return tuple(_state_to_device(item, device) for item in tree)
    if tree is None:
        return None
    raise TypeError(f"Unsupported VPT state node: {type(tree)}")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VPTPolicy(TorchPolicy):
    """Transformer VPT adapter, independent of Ray, Lightning and v1 MinePolicy.

    act runs in evaluation/inference mode. A dedicated training/batch API will be
    added in a later alpha; nn.Module inheritance alone is not that contract.
    """

    def __init__(self, config: VPTConfig | None = None) -> None:
        super().__init__()
        self.config = config or VPTConfig()
        config = self.config
        self.codec = VPTActionCodec()
        self.net = MinecraftPolicy(**config.network_kwargs())
        self.pi_head = make_action_head(
            gym.spaces.Dict(
                {
                    "buttons": gym.spaces.MultiDiscrete([8641]),
                    "camera": gym.spaces.MultiDiscrete([121]),
                }
            ),
            config.hidden_size,
            temperature=config.temperature,
        )
        self.value_head = ScaledMSEHead(config.hidden_size, 1)
        self.eval()

    def initial_state(self) -> VPTState:
        return VPTState(_state_to_device(self.net.initial_state(1), self.device))

    def act(
        self,
        observation: Observation,
        state: VPTState | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[MinecraftAction, VPTState]:
        step, next_state = self.act_with_stats(observation, state, deterministic=deterministic)
        return step.action, next_state

    @torch.inference_mode()
    def act_with_stats(
        self,
        observation: Observation,
        state: VPTState | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[VPTStep, VPTState]:
        if self.training:
            raise RuntimeError("Call eval() before VPT inference")
        state = self.initial_state() if state is None else state
        image = snapshot_observation(observation)["image"]
        size = self.config.image_size
        if image.shape[:2] != (size.height, size.width):
            image = cv2.resize(image, (size.width, size.height), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(image.copy()).to(self.device)[None, None]
        first = torch.tensor([[state.first]], dtype=torch.bool, device=self.device)
        (policy_latent, value_latent), recurrent = self.net(
            {"image": image_tensor},
            _state_to_device(state.recurrent, self.device),
            {"first": first},
        )
        logits = self.pi_head(policy_latent)
        sampled = self.pi_head.sample(logits, deterministic=deterministic)
        encoded: VPTAction = {
            "buttons": int(sampled["buttons"].item()),
            "camera": int(sampled["camera"].item()),
        }
        log_prob = 0.0 if deterministic else float(self.pi_head.logprob(sampled, logits).item())
        value = float(self.value_head.denormalize(self.value_head(value_latent)).item())
        return VPTStep(self.codec.decode(encoded), encoded, log_prob, value), VPTState(
            recurrent, False
        )

    def save_pretrained(
        self, directory: str | Path, *, provenance: dict[str, object] | None = None
    ) -> None:
        """Atomically create a local inference export. Refuse to overwrite anything."""
        destination = Path(directory)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".vpt-", dir=destination.parent))
        try:
            weights = staging / "model.safetensors"
            save_file(
                {
                    name: tensor.detach().cpu().contiguous()
                    for name, tensor in self.state_dict().items()
                },
                str(weights),
            )
            manifest = {
                "schema_version": 1,
                "policy_type": "vpt",
                "minestudio_version": __version__,
                "codec": self.codec.name,
                "processor": "rgb-uint8-cv2-linear-v1",
                "config": asdict(self.config),
                "weights_sha256": _digest(weights),
                "provenance": provenance,
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            )
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    @classmethod
    def from_pretrained(cls, directory: str | Path, *, device: str = "cpu") -> "VPTPolicy":
        """Load only a v2 local manifest + safetensors export; never fetch or unpickle."""
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        expected = {
            "schema_version": 1,
            "policy_type": "vpt",
            "codec": VPTActionCodec.name,
            "processor": "rgb-uint8-cv2-linear-v1",
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Unsupported VPT manifest {key}: {manifest.get(key)!r}")
        weights = directory / "model.safetensors"
        if _digest(weights) != manifest["weights_sha256"]:
            raise ValueError("VPT weights SHA-256 mismatch")
        config = dict(manifest["config"])
        config["image_size"] = ImageSize(**config["image_size"])
        if "channels" in config:
            config["channels"] = tuple(config["channels"])
        policy = cls(VPTConfig(**config))
        policy.load_state_dict(load_file(str(weights)), strict=True)
        return policy.to(device).eval()

    @classmethod
    def from_legacy_state_dict(
        cls, state_dict: Mapping[str, torch.Tensor], *, config: VPTConfig, prefix: str = ""
    ) -> "VPTPolicy":
        """Explicit import of an already loaded v1 state dict with matching config.

        Translate v1 architecture options with VPTConfig.from_legacy_kwargs first. No missing or
        unexpected keys are filtered; strict loading reports any mismatch.
        """
        if any(not name.startswith(prefix) for name in state_dict):
            raise ValueError("Every legacy state-dict key must have the specified prefix")
        policy = cls(config)
        policy.load_state_dict(
            {name[len(prefix) :]: tensor for name, tensor in state_dict.items()}, strict=True
        )
        return policy.eval()
