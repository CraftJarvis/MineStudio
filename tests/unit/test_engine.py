import hashlib
import zipfile

import pytest

from minestudio.core import EngineNotFoundError
from minestudio.envs.engine import engine_root, install_engine, require_engine


@pytest.fixture
def engine_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MINESTUDIO_DIR", str(tmp_path / "cache"))
    return tmp_path


def archive_at(path, member="engine/build/libs/mcprec-6.13.jar"):
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(member, b"fixture")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_engine_discovery_has_no_side_effects(engine_dir):
    assert not engine_root().parent.exists()
    with pytest.raises(EngineNotFoundError):
        require_engine()
    assert not engine_root().parent.exists()


def test_install_validates_hash_and_preserves_existing_install(engine_dir):
    archive = engine_dir / "engine.zip"
    digest = archive_at(archive)
    with pytest.raises(ValueError, match="SHA-256"):
        install_engine(archive, sha256="0" * 64)
    assert not engine_root().exists()
    root = install_engine(archive, sha256=digest)
    assert require_engine() == root
    assert (root / "minestudio-install.json").is_file()
    with pytest.raises(FileExistsError):
        install_engine(archive, sha256=digest)


@pytest.mark.parametrize(
    "member", ["../escape", "/tmp/escape", "engine/../../escape", "elsewhere/file"]
)
def test_unsafe_zip_rejected_without_partial_install(engine_dir, member):
    archive = engine_dir / "engine.zip"
    digest = archive_at(archive, member)
    with pytest.raises(ValueError, match="Unsafe"):
        install_engine(archive, sha256=digest)
    assert not engine_root().exists()
    assert list(engine_root().parent.iterdir()) == []
