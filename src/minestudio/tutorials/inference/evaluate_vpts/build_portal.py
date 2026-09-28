'''
Date: 2024-12-13 14:31:12
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-04 14:06:19
FilePath: /MineStudio/minestudio/tutorials/inference/evaluate_vpts/build_portal.py
'''
import ray
from rich import print
from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter

from functools import partial
from minestudio.models import load_vpt_policy, VPTPolicy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import SpeedTestCallback, CommandsCallback


if __name__ == '__main__':
    ray.init()
    env_generator = partial(
        MinecraftSim, 
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        callbacks=[
            SpeedTestCallback(50), 
            CommandsCallback(commands=[
                '/give @p minecraft:obsidian 64',
            ]), 
        ]
    )
    agent_generator = lambda: VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_for_build_portal_2x")
    worker_kwargs = dict(
        env_generator=env_generator, 
        agent_generator=agent_generator,
        num_max_steps=1200,
        num_episodes=2,
        tmpdir="./output",
        image_media="h264",
    )
    pipeline = EpisodePipeline(
        episode_generator=MineGenerator(
            num_workers=8,
            num_gpus=0.25,
            max_restarts=3,
            **worker_kwargs, 
        ), 
    )
    summary = pipeline.run()
    print(summary)