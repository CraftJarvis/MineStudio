'''
Date: 2024-11-12 14:00:50
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-15 17:02:32
FilePath: /MineStudio/minestudio/tutorials/offline/1_finetune_vpts/train_raw.py
'''
import hydra
import lightning as L

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from minestudio.data import RawDataModule
from minestudio.data.minecraft.callbacks import ImageKernelCallback, VectorActionKernelCallback
from minestudio.models import VPTFlowPolicy
from minestudio.offline import MineLightning
from minestudio.offline.utils import convert_to_normal
from minestudio.offline.mine_callbacks import FlowMatchingCallback
from minestudio.offline.lightning_callbacks import SmartCheckpointCallback, SpeedMonitorCallback
from minestudio.offline.utils import convert_to_normal

logger = WandbLogger(project="minestudio")
@hydra.main(config_path='.', config_name='vpt_flow_config')
def main(args):
    
    mine_policy = VPTFlowPolicy(policy_kwargs=convert_to_normal(args.policy), action_kwargs=convert_to_normal(args.action))
    mine_lightning = MineLightning(
        mine_policy=mine_policy,
        log_freq=20,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        callbacks=[
            FlowMatchingCallback(sigma=args.fm.sigma),
        ], 
        hyperparameters=convert_to_normal(args),
    )

    mine_data = RawDataModule(
        data_params=dict(
            dataset_dirs=args.data.dataset_dirs, 
            modal_kernel_callbacks=[
                ImageKernelCallback(
                    frame_width=args.data.frame_width, 
                    frame_height=args.data.frame_height, 
                    enable_video_aug=False
                ),
                VectorActionKernelCallback(action_chunk_size=args.data.action_chunk_size),
            ],
            win_len=args.data.win_len, 
            split_ratio=args.split_ratio, 
            shuffle_episodes=args.shuffle_episodes,
        ), 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        episode_continuous_batch=args.episode_continuous_batch, 
    )

    L.Trainer(
        logger=logger, 
        devices=args.devices, 
        precision=args.precision, 
        strategy='ddp_find_unused_parameters_true', 
        gradient_clip_val=1.0, 
        use_distributed_sampler=not args.episode_continuous_batch,
        callbacks=[
            LearningRateMonitor(logging_interval='step'), 
            SpeedMonitorCallback(),
            SmartCheckpointCallback(
                dirpath='./weights', filename='weight-{epoch}-{step}', save_top_k=-1, 
                every_n_train_steps=args.save_freq, save_weights_only=True,
            ), 
            SmartCheckpointCallback(
                dirpath='./checkpoints', filename='ckpt-{epoch}-{step}', save_top_k=1, 
                every_n_train_steps=args.save_freq+1, save_weights_only=False,
            )
        ]
    ).fit(
        model=mine_lightning, 
        datamodule=mine_data
    )

if __name__ == '__main__':
    main()
