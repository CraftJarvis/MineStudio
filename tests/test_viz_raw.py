import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from typing import Union, Tuple, List, Dict, Callable, Sequence, Mapping, Any, Optional

from minestudio.data import RawDataset
from minestudio.data.minecraft.utils import (
    MineDistributedBatchSampler, write_video, batchify, visualize_dataloader
)
from minestudio.data.minecraft.callbacks import (
    ImageKernelCallback, 
    ActionKernelCallback, 
    MetaInfoKernelCallback, 
    SegmentationKernelCallback, 
    ActionDrawFrameCallback, 
    MetaInfoDrawFrameCallback, 
    SegmentationDrawFrameCallback
)

def visualize_raw_dataset(args):
    raw_dataset = RawDataset(
        dataset_dirs=args.dataset_dirs, 
        modal_kernel_callbacks=[
            ImageKernelCallback(
                frame_width=args.frame_width, 
                frame_height=args.frame_height, 
                enable_video_aug=args.enable_video_aug,
            ), 
            ActionKernelCallback(enable_prev_action=True, win_bias=1, read_bias=-1),
            MetaInfoKernelCallback(),
            SegmentationKernelCallback(
                frame_width=args.frame_width, 
                frame_height=args.frame_height, 
            )
        ],
        win_len=args.win_len, 
        skip_frame=args.skip_frame,
        shuffle_episodes=args.shuffle_episodes,
    )
    Console().log(f"num-workers: {args.num_workers}")
    batch_sampler = MineDistributedBatchSampler(
        dataset=raw_dataset,
        batch_size=args.batch_size,
        num_replicas=1, 
        rank=0,
    )
    dataloader = DataLoader(
        dataset=raw_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=batchify,
    )
    
    visualize_dataloader(
        dataloader, 
        draw_frame_callbacks=[
            ActionDrawFrameCallback(), 
            MetaInfoDrawFrameCallback(), 
            SegmentationDrawFrameCallback()
        ],
        num_samples=args.num_samples, 
        save_fps=args.save_fps,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dirs', type=str, nargs='+', required=True)
    parser.add_argument('--win-len', type=int, default=128)
    parser.add_argument('--skip-frame', type=int, default=1)
    parser.add_argument('--frame-width', type=int, default=224)
    parser.add_argument('--frame-height', type=int, default=224)
    parser.add_argument('--enable-video-aug', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--shuffle-episodes', action='store_true', default=False)
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--save-fps', type=int, default=30)
    args = parser.parse_args()
    visualize_raw_dataset(args)