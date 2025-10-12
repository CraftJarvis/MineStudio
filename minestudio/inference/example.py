'''
Date: 2024-11-14 19:42:09
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-15 13:36:22
FilePath: /MineStudio/minestudio/inference/example.py
'''

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback, CommandsCallback
from minestudio.models import VPTPolicy, load_vpt_policy

if __name__ == '__main__':
    while True:
        
        env = MinecraftSim(
            action_type='env',
            obs_size=(128, 128), 
            preferred_spawn_biome="plains", 
            callbacks=[
                #RecordCallback(record_path=f'output', fps=30, frame_type="pov"),
                CommandsCallback(
                    ["/setblock ~ ~ ~3 minecraft:crafting_table"]
                )
            ]
        )
        env.reset()
        for i in range(20):
            print("Step:", i)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        env.close()
        del env