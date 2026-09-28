'''
Date: 2024-12-06 16:42:49
LastEditors: zhengxinyue
LastEditTime: 2024-12-12 17:44:10
FilePath: /MineStudio/minestudio/benchmark/test_pipeline.py
'''
import os
from pathlib import Path
from functools import partial
from minestudio.simulator import MinecraftSim
from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter
from minestudio.benchmark.utility.read_conf import convert_yaml_to_callbacks
from minestudio.benchmark.utility.task_call import TaskCallback
from minestudio.models import VPTPolicy, load_vpt_policy
from minestudio.simulator.callbacks import (
    RecordCallback, 
    RewardsCallback, 
    CommandsCallback, 
)

if __name__ == '__main__':
    conf_path = './task_configs/simple'
    
    for file_name in os.listdir(conf_path):
        if file_name.endswith('.yaml'):
            file_path = os.path.join(conf_path, file_name)
            commands_callback, task_callback = convert_yaml_to_callbacks(file_path)
            print(f'Task: {task_callback}')
            print(f'Init commands: {commands_callback}')

            folder_path = Path(f'{file_name[:-5]}')
            folder_path.mkdir(exist_ok=True)

            env_generator = partial(
                MinecraftSim, 
                obs_size=(128, 128), 
                callbacks=[
                    RecordCallback(record_path=f"./output/{folder_path}/", fps=30, frame_type="pov"),
                    CommandsCallback(commands_callback),
                    TaskCallback(task_callback),
                ]
            )

            agent_generator = partial(
                load_vpt_policy,
                model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
                weights_path="/nfs-shared/jarvisbase/pretrained/rl-from-early-game-2x.weights"
            )
            worker_kwargs = dict(
                env_generator=env_generator, 
                agent_generator=agent_generator,
                num_max_steps=200,
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
                )
            )
            summary = pipeline.run()
            print(summary)