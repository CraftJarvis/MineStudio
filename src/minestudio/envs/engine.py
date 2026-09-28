"""Explicit engine discovery and local archive installation (no implicit download)."""

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from minestudio.core import EngineNotFoundError

_MARKER = Path("build/libs/mcprec-6.13.jar")


def engine_root() -> Path:
    """Resolve the legacy-compatible MINESTUDIO_DIR without creating directories."""
    return (
        Path(os.environ.get("MINESTUDIO_DIR", str(Path(tempfile.gettempdir()) / "MineStudio")))
        / "engine"
    )


def require_engine() -> Path:
    root = engine_root()
    if not (root / _MARKER).is_file():
        raise EngineNotFoundError(
            f"Minecraft engine missing at {root}. Install explicitly with "
            "`minestudio engine install ARCHIVE --sha256 SHA256`; see the installation guide."
        )
    return root


def install_engine(archive: Path, *, sha256: str) -> Path:
    """Verify a local engine.zip, reject unsafe members, then install atomically.

    The archive must contain engine/build/libs/mcprec-6.13.jar. Existing installs
    are never overwritten. Obtain the expected digest from a trusted source.
    """
    target = engine_root()
    if target.exists():
        raise FileExistsError(f"Engine destination already exists: {target}")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256.lower():
        raise ValueError("Engine archive SHA-256 mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".engine-", dir=target.parent))
    try:
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                member = Path(entry.filename)
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or not member.parts
                    or member.parts[0] != "engine"
                    or "\\" in entry.filename
                    or stat.S_ISLNK(entry.external_attr >> 16)
                ):
                    raise ValueError(f"Unsafe archive member: {entry.filename}")
            bundle.extractall(staging)
            for entry in bundle.infolist():
                extracted = staging / entry.filename
                if extracted.is_file() and (entry.external_attr >> 16) & 0o111:
                    extracted.chmod(0o755)
        extracted_root = staging / "engine"
        if not (extracted_root / _MARKER).is_file():
            raise ValueError("Archive does not contain the required Minecraft engine")
        (extracted_root / "minestudio-install.json").write_text(
            json.dumps({"schema_version": 1, "archive_sha256": digest.hexdigest()}, indent=2) + "\n"
        )
        extracted_root.rename(target)
    finally:
        shutil.rmtree(staging)
    return target
