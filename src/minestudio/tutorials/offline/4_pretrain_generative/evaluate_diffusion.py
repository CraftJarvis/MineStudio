'''
Date: 2025-01-18 23:15:39
LastEditors: muzhancun 2100017790@stu.pku.edu.cn
LastEditTime: 2025-01-27 13:56:37
FilePath: /MineStudio/minestudio/tutorials/offline/4_pretrain_generative/evaluate_diffusion.py
'''
import numpy as np
from minestudio.simulator import MinecraftSim
from functools import partial
from minestudio.models.vpt_diffusion import VPTDiffusionPolicy, load_vpt_diffusion_policy
from minestudio.simulator.callbacks import MinecraftCallback, RecordCallback
import torch

class NoiseCallback(MinecraftCallback):
    def __init__(self, action_shape=220):
        super().__init__()
        self.action_shape = action_shape

    def after_reset(self, sim, obs, info):
        obs['noise'] = np.random.randn(self.action_shape).astype(np.float32)
        # print(obs['noise'].shape)
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        obs['noise'] = np.random.randn(self.action_shape).astype(np.float32)
        return obs, reward, terminated, truncated, info

if __name__ == '__main__':
    import os
    dir = ""
    # find the latest checkpoint
    checkpoints = os.listdir(dir)
    checkpoints = [os.path.join(dir, c) for c in checkpoints if c.endswith("ckpt")]
    checkpoints.sort(key=os.path.getmtime)
    print(checkpoints)
    agent = load_vpt_diffusion_policy(checkpoints[-1])
    agent = agent.to('cuda:0')

    sim = MinecraftSim(
        obs_size=(128, 128),
        callbacks=[
            RecordCallback(record_path='./output', recording=True),
            NoiseCallback(action_shape=220),
        ],
        action_type = 'env',
        preferred_spawn_biome = "plains", 
    )

    obs, info = sim.reset()

    memory = None

    for i in range(100):
        action, memory = agent.get_action(obs, memory, input_shape='*')
        print(action)
        # action["camera"][0] = np.clip(action["camera"][0], -20.0, 20.0)
        # action["camera"][1] = np.clip(action["camera"][1], -20.0, 20.0)
        # action["camera"][0] = 0.0
        # action["camera"][1] = 0.0
        print(obs["image"].shape, obs["noise"].shape)
        obs, reward, terminated, truncated, info = sim.step(action)
    sim.close()