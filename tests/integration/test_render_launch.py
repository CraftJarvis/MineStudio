"""Reject VirtualGL failures before Java silently starts with software OpenGL."""

import os
import shutil
import subprocess
from importlib.resources import files

import pytest


@pytest.mark.parametrize("renderer", ["NVIDIA Corporation", "Mesa", "failure"])
def test_gpu_launch_checks_actual_renderer(tmp_path, renderer):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Minecraft launch script requires bash")
    scripts = {
        "vglrun": '#!/bin/sh\nshift 2\nexec "$@"\n',
        "glxinfo": (
            '#!/bin/sh\n[ "$TEST_RENDERER" = failure ] && exit 1\n'
            'printf "OpenGL vendor string: %s\\n" "$TEST_RENDERER"\n'
        ),
        "java": '#!/bin/sh\nprintf "%s\\n" "$@" > "$JAVA_ARGS"\n',
    }
    for name, content in scripts.items():
        script = tmp_path / name
        script.write_text(content)
        script.chmod(0o755)
    arguments = tmp_path / "java-args"
    result = subprocess.run(
        [
            bash,
            str(files("minestudio") / "simulator/minerl/env/launchClient.sh"),
            "-device",
            "/dev/dri/card1",
            "-fatjar",
            "game with spaces.jar",
            "-port",
            "12345",
        ],
        env=dict(
            os.environ,
            PATH=f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
            TEST_RENDERER=renderer,
            JAVA_ARGS=str(arguments),
        ),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if renderer == "NVIDIA Corporation":
        assert result.returncode == 0, result.stderr
        assert arguments.read_text().splitlines() == [
            "-Xmx2G",
            "-jar",
            "game with spaces.jar",
            "--envPort=12345",
        ]
    else:
        assert result.returncode != 0
        assert not arguments.exists()
