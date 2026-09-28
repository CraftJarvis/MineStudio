from minestudio.simulator.callbacks import MinecraftCallback
from minestudio.models import SteveOnePolicy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import SpeedTestCallback, load_callbacks_from_config
from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter
from minestudio.benchmark import prepare_task_configs

import ray
from functools import partial
from rich import print

class CommandCallback(MinecraftCallback):
    """
    To use SteveOnePolicy, you need to contain a condition in the observation.
    """
    def __init__(self, command, cond_scale = 4.0):
        self.command = command
        self.cond_scale = cond_scale

    def after_reset(self, sim, obs, info):
        self.timestep = 0
        obs["condition"] = {
            "cond_scale": self.cond_scale,
            "text": self.command
        }
        return obs, info
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        obs["condition"] = {
            "cond_scale": self.cond_scale,
            "text": self.command
        }
        return obs, reward, terminated, truncated, info


if __name__ == '__main__':
    ray.init()
    task_configs = prepare_task_configs("simple", path="CraftJarvis/MineStudio_task_group.simple")
    config_file = task_configs["collect_wood"] 
    # you can try: survive_plant, collect_wood, build_pillar, ... ; make sure the config file contains `reference_video` field 
    print(config_file)

    env_generator = partial(
        MinecraftSim,
        obs_size = (128, 128),
        preferred_spawn_biome = "forest", 
        callbacks = [
            SpeedTestCallback(50),
            CommandCallback("mine log", cond_scale=4.0),  # Add a command callback for SteveOnePolicy
        ] + load_callbacks_from_config(config_file)
    )

    agent_generator = lambda: SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official")

    worker_kwargs = dict(
        env_generator=env_generator, 
        agent_generator=agent_generator,
        num_max_steps=600,
        num_episodes=1,
        tmpdir="./output",
        image_media="h264",
    )

    pipeline = EpisodePipeline(
        episode_generator=MineGenerator(
            num_workers=1, 
            num_gpus=0.25,
            max_restarts=3,
            **worker_kwargs, 
        ), 
        episode_filter=InfoBaseFilter(
            key="mine_block",
            regex=".*log.*",
            num=1,
        ),
    )
    summary = pipeline.run()
    print(summary)