# Contributing to MineStudio v2

Install from the repository root (Python 3.10–3.12). Lightweight development does
not require the game or Torch:

```bash
python -m venv .venv-stage2
source .venv-stage2/bin/activate
python -m pip install -e '.[dev]' gymnasium
ruff check .
ruff format --check .
pyright
python -m pytest
python examples/v2/task_evaluation.py --output-dir runs/task-demo
python -m build
```

`.venv-stage2` is the local environment used by Pyright; when using another one,
pass `pyright --pythonpath /path/to/python`. Install optional dependencies for the
affected areas, then run their checks:

```bash
python -m pip install -e '.[envs,policies,docs]'
python -m pytest tests/integration/test_vpt.py tests/integration/test_minerl_import.py -q
python examples/v2/task_evaluation.py --output-dir runs/task-video --video
mkdocs build --strict
```

The new public contract is defined in [the architecture specification](docs/architecture/v2.md).
New code uses explicit domain imports, typed public functions, Google-style docstrings,
100-column Ruff formatting and caller-owned policy state. Optional runtimes belong
behind explicit feature imports. Configuration errors must remain visible.

Pyright strict currently covers core/actions/envs/policy protocols/VPT config/rollout/CLI,
tasks, native trajectory storage and evaluation. The
MineRL boundary suppresses unknown types from legacy libraries; VPT's copied network
and old namespaces are not in this initial strict coverage. Expand coverage as those
implementations migrate. Ruff scopes are explicit in pyproject.toml. The native
trajectory reader and evaluation modules must remain importable with NumPy alone.

Tests under tests/unit and tests/integration are automated. Existing root-level test
scripts are historical manual experiments and intentionally excluded from collection.
Real engine tests use the engine marker and live under tests/e2e. Policy tests require optional Torch
packages and have a dedicated CI job; a skipped runtime test is not validation.
With engine/display dependencies and a verified VPT export available, run:

```bash
MINESTUDIO_RUN_ENGINE=1 MINESTUDIO_VPT_EXPORT=/path/to/vpt-export python -m pytest tests/e2e -q
```

The export variable is required for the second e2e test; without it only the engine
test runs. FFmpeg and ffprobe are required for video checks. See the
[validation record](docs/architecture/stage2-validation.md) for what actually ran locally.

Preview documentation with `mkdocs serve`. Write task-oriented guides with one
primary route, expected output and troubleshooting. Keep planned and implemented APIs
clearly distinguished. The old Sphinx sources remain available for v1 reference.

Builds and documentation previews are local until explicitly published. New CI builds
artifacts and does not deploy the v2 site.
