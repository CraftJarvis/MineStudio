import lightning as L
from tqdm import tqdm

from minestudio.data import RawDataModule
from minestudio.data.minecraft.callbacks import (
    ImageKernelCallback, ActionKernelCallback, SegmentationKernelCallback
)

continuous_batch = True

fabric = L.Fabric(accelerator="cuda", devices=2, strategy="ddp")
fabric.launch()
data_module = RawDataModule(
    data_params=dict(
        dataset_dirs=[
            '/nfs-shared-2/data/contractors-new/dataset_6xx',
            '/nfs-shared-2/data/contractors-new/dataset_7xx',
            '/nfs-shared-2/data/contractors-new/dataset_8xx',
            '/nfs-shared-2/data/contractors-new/dataset_9xx',
            '/nfs-shared-2/data/contractors-new/dataset_10xx',
        ],
        modal_kernel_callbacks=[
            ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False),
            ActionKernelCallback(),
            SegmentationKernelCallback(frame_width=224, frame_height=224),
        ],
        win_len=128,
        split_ratio=0.9,
        shuffle_episodes=True,
    ),
    batch_size=4,
    num_workers=2,
    prefetch_factor=4,
    episode_continuous_batch=continuous_batch, 
)
data_module.setup()
train_loader = data_module.train_dataloader()
train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=not continuous_batch)
rank = fabric.local_rank
for idx, batch in enumerate(tqdm(train_loader, disable=True)):
    if idx > 50:
        break
    ...
    print(
        f"{rank = } \t" + "\t".join(
            [f"{a[-20:]} {b}" for a, b in zip(batch['episode'], batch['progress'])]
        )
    )
