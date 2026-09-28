# MineStudio

Composable tools for Minecraft agent research.

**v2 is under development (`2.0.0a2`).** Define tasks, run local evaluations, and save
checked trajectories and optional MP4 videos through explicit environment and policy
interfaces. The MineRL and VPT adapters remain experimental; Linux CPU game smoke tests
and VPT foundation 1x weight parity now pass. Other platforms and model variants need validation. See the [implementation status](docs/architecture/progress.md)
and [stage-two validation record](docs/architecture/stage2-validation.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . gymnasium
minestudio doctor
python examples/v2/task_evaluation.py --output-dir runs/task-demo
```

This example uses a synthetic counter backend. It evaluates two seeded cases and
reads back each recorded trajectory; add `--video` with FFmpeg installed to save MP4s.
It needs neither Minecraft nor Torch. Real Minecraft requires an explicitly installed
engine, Java 8 and a display/rendering setup; see [installation](docs/getting-started/installation.md).

| Start here | Purpose |
| --- | --- |
| [Quickstart](docs/getting-started/quickstart.md) | Run an evaluation and inspect its artifacts |
| [Tasks](docs/guides/tasks.md) | Define initialization, rewards and end conditions |
| [Evaluation](docs/guides/evaluation.md) | Fix task/seed cases and understand reported metrics |
| [Trajectory storage](docs/guides/trajectories.md) | Read, verify and recover recorded attempts |
| [Architecture specification](docs/architecture/v2.md) | Terminology, module boundaries, APIs and code conventions |
| [Migration guide](docs/migration/v1-to-v2.md) | Understand compatibility and the current alpha limits |
| [Contributing](CONTRIBUTING.md) | Development setup, checks and documentation preview |
| [v1 documentation](docs/legacy/README-v1.md) | Original project introduction and workflows |

To preview the new documentation locally:

```bash
python -m pip install -e '.[docs]'
mkdocs serve
```

MineStudio is developed by [CraftJarvis](https://craftjarvis.github.io/).
[Paper](https://arxiv.org/abs/2412.18293) ·
[Models and datasets](https://huggingface.co/CraftJarvis) · [MIT license](LICENSE)
