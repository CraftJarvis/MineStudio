import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from typing import Union, Tuple, List, Dict, Callable, Sequence, Mapping, Any, Optional

from minestudio.data import EventDataset
from minestudio.data.minecraft.utils import (
    MineDistributedBatchSampler, write_video, batchify, visualize_dataloader
)
from minestudio.data.minecraft.callbacks import (
    ImageKernelCallback, ActionKernelCallback, MetaInfoKernelCallback, SegmentationKernelCallback, 
    ActionDrawFrameCallback, MetaInfoDrawFrameCallback, SegmentationDrawFrameCallback
)

def visualize_event_dataset(args):
    event_dataset = EventDataset(
        dataset_dirs=args.dataset_dirs, 
        modal_kernel_callbacks=[
            ImageKernelCallback(
                frame_width=args.frame_width, 
                frame_height=args.frame_height, 
                enable_video_aug=args.enable_video_aug,
            ), 
            ActionKernelCallback(),
            MetaInfoKernelCallback(),
            SegmentationKernelCallback(
                frame_width=args.frame_width, 
                frame_height=args.frame_height, 
            )
        ],
        win_len=args.win_len, 
        skip_frame=args.skip_frame,
        event_regex=args.event_regex,
        min_nearby=args.min_nearby,
        max_within=args.max_within,
    )
    Console().log(f"num-workers: {args.num_workers}")
    dataloader = DataLoader(
        dataset=event_dataset,
        batch_size=args.batch_size,
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
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--save-fps', type=int, default=30)
    parser.add_argument('--event-regex', type=str, default='minecraft.mine_block:.*diamond.*')
    parser.add_argument('--min-nearby', type=int, default=64)
    parser.add_argument('--max-within', type=int, default=1000)
    args = parser.parse_args()
    visualize_event_dataset(args)