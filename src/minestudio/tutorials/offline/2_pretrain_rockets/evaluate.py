'''
Date: 2024-12-17 02:07:48
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-17 06:14:38
FilePath: /MineStudio/minestudio/tutorials/offline/2_pretrain_rockets/evaluate.py
'''

import hydra
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from einops import rearrange
from typing import Dict, Any, Tuple

import minestudio.models
from minestudio.utils.register import Registers
from minestudio.data import MineDataModule
from minestudio.offline import MineLightning
from minestudio.offline.mine_callbacks import BehaviorCloneCallback

if __name__ == '__main__':
    
    policy_loader = Registers.model_loader['load_rocket_policy']
    mine_policy = policy_loader(
        ckpt_path="/nfs-shared-2/shaofei/minestudio/save/2024-12-10/11-02-34/weights/weight-epoch=1-step=30000.ckpt", 
    )
    
    mine_lightning = MineLightning(
        mine_policy=mine_policy, 
        callbacks=[
            BehaviorCloneCallback(weight=0.01),
        ], 
    )
    
    mine_data = MineDataModule(
        data_params=dict(
            mode='raw',
            dataset_dirs=[
                '/nfs-shared-2/data/contractors/dataset_6xx',
                '/nfs-shared-2/data/contractors/dataset_7xx',
                '/nfs-shared-2/data/contractors/dataset_8xx',
                '/nfs-shared-2/data/contractors/dataset_9xx',
                '/nfs-shared-2/data/contractors/dataset_10xx'
            ],
            frame_width=224,
            frame_height=224,
            win_len=128,
            enable_segment=True,
        ),
        batch_size=8,
        num_workers=4,
        prefetch_factor=4,
        split_ratio=0.90, 
        shuffle_episodes=True,
        episode_continuous_batch=False,
    )
    
    # evaluate the model
    mine_lightning.eval()
    mine_lightning.freeze()
    L.Trainer(
        precision="bf16",
        strategy='ddp_find_unused_parameters_true', 
        devices=8, 
        use_distributed_sampler=True,
    ).validate(mine_lightning, mine_data)
    