'''
Date: 2024-12-06 16:35:39
LastEditors: zhengxinyue
LastEditTime: 2024-12-11 17:47:25
FilePath: /MineStudio/minestudio/benchmark/test.py
'''

import os
import ray
from pathlib import Path
from rich import print
from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter
from minestudio.benchmark.utility.read_conf import convert_yaml_to_callbacks
# from minestudio.benchmark.utility.record_call import RecordCallback
from minestudio.benchmark.utility.task_call import TaskCallback
from functools import partial
from minestudio.models import load_vpt_policy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RewardsCallback, CommandsCallback, RecordCallback

if __name__ == '__main__':
    ray.init()
    conf_path = './task_configs/simple'
    
    for file_name in os.listdir(conf_path):
        if file_name.endswith('.yaml'):
            file_path = os.path.join(conf_path, file_name)
            commands_callback, task_callback = convert_yaml_to_callbacks(file_path)
            print(f'Task: {task_callback}')
            print(f'Init commands: {commands_callback}')

            folder_path = Path(f'{file_name[:-5]}')
            folder_path.mkdir(exist_ok=True)

            env = MinecraftSim(
                obs_size=(128, 128), 
                callbacks=[
                    RecordCallback(record_path=f"./output/{folder_path}/", fps=30, frame_type="pov"),
                    CommandsCallback(commands_callback),
                    TaskCallback(task_callback),
                ]
            )
            policy = load_vpt_policy(
                model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
                weights_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.weights"
            ).to("cuda")
            
            memory = None
            obs, info = env.reset()
            print('New rollout begins')
            for i in range(12):
                action, memory = policy.get_action(obs, memory, input_shape='*')
                obs, reward, terminated, truncated, info = env.step(action)
            
            env.close()

