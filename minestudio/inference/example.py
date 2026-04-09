'''
Date: 2024-11-14 19:42:09
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-15 13:36:22
FilePath: /MineStudio/minestudio/inference/example.py
'''

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback, WorldCheckpointCallback
from minestudio.models import VPTPolicy, load_vpt_policy

if __name__ == '__main__':
    
    policy = load_vpt_policy(
        model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
        weights_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.weights"
    ).to("cuda")

    ckpt_cb = WorldCheckpointCallback(
        export_root="/scratch/hekaichen/checkpoints/minecraft_worlds",
        auto_export_on_close=True,
        export_every_n_steps=None,
        overwrite=True,
        verbose=True,
    ) #现在版本里这些参数基本就auto export on close 设置成true会存checkpoint，其他都待实现...实际上的checkpoint路径在log里查 “[IOWorker]” 这个字符串就找到了，world结尾。
    
    env = MinecraftSim(
        obs_size=(128, 128), 
        #preferred_spawn_biome="forest",
        callbacks=[
            RecordCallback(record_path="./output", fps=30, frame_type="pov"),
            SpeedTestCallback(50),
            ckpt_cb,
        ],
        restore_checkpoint_path="/tmp/4785ceae4906/saves/world" #"/nfs-shared-2/hekaichen/workspace/tmp/world_fix1", 这里目录下直接包含advancements  data  datapacks  DIM-1  DIM1  icon.png  level.dat  level.dat_old  playerdata  poi  region  session.lock  stats 
    )

    memory = None
    obs, info = env.reset()

    for i in range(100):
        action, memory = policy.get_action(obs, memory, input_shape='*')
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()