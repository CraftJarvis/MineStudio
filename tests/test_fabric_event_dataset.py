import lightning as L
from tqdm import tqdm
from minestudio.data import EventDataModule
from minestudio.data.minecraft.callbacks import (
    ImageKernelCallback, ActionKernelCallback, SegmentationKernelCallback
)

fabric = L.Fabric(accelerator="cuda", devices=2, strategy="ddp")
fabric.launch()
data_module = EventDataModule(
    data_params=dict(
        dataset_dirs=[
            '/nfs-shared-2/data/contractors-new/dataset_6xx',
            '/nfs-shared-2/data/contractors-new/dataset_7xx',
        ],
        modal_kernel_callbacks=[
            ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False),
            ActionKernelCallback(),
            SegmentationKernelCallback(frame_width=224, frame_height=224),
        ], 
        win_len=128,
        split_ratio=0.9,
        event_regex='minecraft.mine_block:.*log.*',
    ),
    batch_size=3,
    num_workers=2,
    prefetch_factor=4
)
data_module.setup()
train_loader = data_module.train_dataloader()
train_loader = fabric.setup_dataloaders(train_loader)
rank = fabric.local_rank
for idx, batch in enumerate(tqdm(train_loader, disable=True)):
    if idx > 50:
        break
    print(
        f"{rank = } \t" + "\t".join(
            [f"{a.shape} {b}" for a, b in zip(batch['image'], batch['text'])]
        )
    )
