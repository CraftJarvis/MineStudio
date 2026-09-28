'''
Date: 2024-12-13 22:39:49
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-02-13 20:00:03
FilePath: /MineStudio/minestudio/tutorials/inference/evaluate_groot/main.py
'''
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import SpeedTestCallback, load_callbacks_from_config
from minestudio.models import GrootPolicy, load_groot_policy
from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter
from minestudio.benchmark import prepare_task_configs

import ray
import numpy as np
import av
import os
from functools import partial
from rich import print

if __name__ == '__main__':
    ray.init()
    task_configs = prepare_task_configs("simple", path="CraftJarvis/MineStudio_task_group.simple")
    config_file = task_configs["collect_wood"] 
    # you can try: survive_plant, collect_wood, build_pillar, ... ; make sure the config file contains `reference_video` field 
    print(config_file)

    env_generator = partial(
        MinecraftSim,
        obs_size = (224, 224),
        preferred_spawn_biome = "forest", 
        callbacks = [
            SpeedTestCallback(50),
        ] + load_callbacks_from_config(config_file)
    )

    agent_generator = lambda: GrootPolicy.from_pretrained("CraftJarvis/MineStudio_GROOT.18w_EMA")

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