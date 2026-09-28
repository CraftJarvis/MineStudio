"""Read explicitly referenced v1 chunks without executing arbitrary pickle globals."""

import io
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from minestudio.core import MissingDependencyError


class _ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unsupported legacy pickle global: {module}.{name}")
        return allowed[module, name]


def read_chunks(
    root: Path, reference: dict[str, Any], start: int, stop: int, *, images: bool
) -> Any:
    """Return a contiguous range using the legacy same-index image/action convention.

    Short-lived handles avoid LMDB's duplicate-open restriction across datasets.
    Inputs are immutable local snapshots; lock=False must not be used with writers.
    """
    try:
        import av
        import lmdb
    except ImportError as error:
        raise MissingDependencyError("Install minestudio[data] to read legacy LMDB") from error
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("LMDB reference escapes dataset root")
    size = reference["chunk_size"]
    if size <= 0 or not 0 <= start < stop <= reference["num_frames"]:
        raise ValueError("Invalid legacy chunk range")
    parts = []
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:

            def metadata(key):
                payload = txn.get(key)
                if payload is None:
                    raise ValueError(f"Missing LMDB metadata: {key!r}")
                return _ArrayUnpickler(io.BytesIO(payload)).load()

            if metadata(b"__chunk_size__") != size:
                raise ValueError("Manifest chunk size disagrees with LMDB")
            infos = metadata(b"__chunk_infos__")
            matches = [row for row in infos if row["episode_idx"] == reference["episode_idx"]]
            if len(matches) != 1 or any(
                matches[0][key] != reference[key] for key in ("episode", "num_frames")
            ):
                raise ValueError("Manifest episode identity/length disagrees with LMDB")
            for offset in range(start // size * size, stop, size):
                payload = txn.get(str((reference["episode_idx"], offset)).encode())
                if payload is None:
                    raise ValueError(f"Missing legacy chunk: {path.name}:{offset}")
                if images:
                    with av.open(io.BytesIO(payload)) as video:
                        part = np.stack(
                            [f.to_ndarray(format="rgb24") for f in video.decode(video=0)]
                        )
                    if len(part) != size:
                        raise ValueError("Legacy video chunk has the wrong frame count")
                else:
                    part = _ArrayUnpickler(io.BytesIO(payload)).load()
                    if not isinstance(part, dict) or any(len(v) != size for v in part.values()):
                        raise ValueError("Legacy action chunk has the wrong shape")
                lo, hi = max(start - offset, 0), min(stop - offset, size)
                parts.append(part[lo:hi] if images else {k: v[lo:hi] for k, v in part.items()})
    finally:
        env.close()
    if images:
        return np.concatenate(parts)
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
