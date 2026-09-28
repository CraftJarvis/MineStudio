"""Opt-in real-game acceptance: MINESTUDIO_RUN_ENGINE=1 pytest tests/e2e -q."""

import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        os.environ.get("MINESTUDIO_RUN_ENGINE") != "1", reason="explicit real-game opt-in required"
    ),
]


def test_real_minerl_reset_step_close():
    from minestudio.actions import noop_action
    from minestudio.envs import EnvConfig, MinecraftEnv

    with MinecraftEnv(EnvConfig(max_episode_steps=3)) as env:
        observation, info = env.reset(seed=42)
        assert env.observation_space.contains(observation)
        assert info["engine"]["jar_sha256"]
        for _ in range(3):
            observation, _, terminated, truncated, _ = env.step(noop_action())
            assert env.observation_space.contains(observation)
            if terminated or truncated:
                break
        assert terminated or truncated
    assert env._backend is None


def test_real_vpt_task_artifacts(tmp_path):
    export = os.environ.get("MINESTUDIO_VPT_EXPORT")
    if export is None:
        pytest.skip("MINESTUDIO_VPT_EXPORT must name a verified local v2 export")
    from minestudio.data.storage import TrajectoryReader
    from minestudio.data.storage.trajectory import sha256_file
    from minestudio.envs import MinecraftEnv
    from minestudio.evaluation import EvaluationConfig, EvaluationSuite, evaluate
    from minestudio.policies.vpt import VPTPolicy
    from minestudio.tasks import load_task

    device = os.environ.get("MINESTUDIO_POLICY_DEVICE", "cpu")

    def policy_factory():
        policy = VPTPolicy.from_pretrained(export, device=device)
        assert str(policy.device) == device
        return policy

    result = evaluate(
        policy_factory,
        suite=EvaluationSuite("engine-smoke", "1", (load_task("collect_oak_log"),), (42,)),
        env_factory=lambda *, task: MinecraftEnv(task=task),
        config=EvaluationConfig(
            str(export),
            sha256_file(Path(export) / "manifest.json"),
            "rgb-uint8-cv2-linear-v1",
            "vpt-mu-law-v1",
            output_dir=tmp_path,
            max_episode_steps=3,
            record_video=True,
        ),
    )
    assert result.completed == 1 and result.failed == 0
    case = result.cases[0]
    recorded = TrajectoryReader(tmp_path / result.run_id / case.context.attempt_id).read()
    episode = recorded.as_episode()
    assert len(episode.observations) == len(episode.transitions) + 1
    assert (tmp_path / result.run_id / "videos" / f"{case.context.attempt_id}.mp4").is_file()
