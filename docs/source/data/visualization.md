# Visualization

We provide a utility function `visualize_dataloader` that allows users to generate videos from the dataloader's output. This is useful for debugging, verifying data correctness, and understanding what the model will see.

## `visualize_dataloader` Arguments

Here are the key arguments for the `visualize_dataloader` function:

| Arguments              | Description                                                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataloader`           | A PyTorch `DataLoader` instance.                                                                                                                                            |
| `draw_frame_callbacks` | A list of `DrawFrameCallback` instances (e.g., `ActionDrawFrameCallback`, `MetaInfoDrawFrameCallback`). These callbacks overlay additional information onto the video frames. |
| `num_samples`          | Number of batches to visualize from the dataloader.                                                                                                                         |
| `save_fps`             | Integer. Frames per second for the saved video.                                                                                                                             |
| `output_dir`           | (Optional) String. Path to the directory where the output videos will be saved. Defaults to a sub-directory like `output_videos` in the current working directory.        |

## Visualize Continuous Batches (Using `RawDataset`)

To visualize continuous segments of data (e.g., sequences from an episode), you can use `RawDataset`. The `RawDataset` loads windows of multi-modal data. The example below shows how to set it up and visualize its output.

```python
import argparse
from torch.utils.data import DataLoader

from minestudio.data import RawDataset
from minestudio.data.minecraft.utils import (
    MineDistributedBatchSampler, batchify, visualize_dataloader
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

# Configuration (similar to args in test_viz_raw.py)
dataset_dirs = ['/nfs-shared-2/data/contractors/dataset_6xx'] # Replace with your dataset path(s)
win_len = 128
skip_frame = 1
frame_width = 224
frame_height = 224
enable_video_aug = False # Or True
batch_size = 1 # Number of sequences per batch
num_workers = 2
shuffle_episodes = False # For continuous visualization, typically False or handle order later
num_samples_to_viz = 3 # Number of video samples to generate
save_fps_viz = 20

# 1. Define Modal Kernel Callbacks for data loading
modal_kernel_callbacks = [
    ImageKernelCallback(
        frame_width=frame_width, 
        frame_height=frame_height, 
        enable_video_aug=enable_video_aug,
    ), 
    ActionKernelCallback(enable_prev_action=True, win_bias=1, read_bias=-1), # Example config
    MetaInfoKernelCallback(),
    SegmentationKernelCallback( # If you have segmentation data
        frame_width=frame_width, 
        frame_height=frame_height, 
    )
]

# 2. Create RawDataset
raw_dataset = RawDataset(
    dataset_dirs=dataset_dirs, 
    modal_kernel_callbacks=modal_kernel_callbacks,
    win_len=win_len, 
    skip_frame=skip_frame,
    shuffle_episodes=shuffle_episodes,
)

# 3. Create DataLoader
# For RawDataset, especially if order matters or for specific sampling:
batch_sampler = MineDistributedBatchSampler( # Simplified for single-process visualization
    dataset=raw_dataset,
    batch_size=batch_size,
    num_replicas=1, 
    rank=0,
    shuffle=shuffle_episodes # Sampler's shuffle
)
dataloader = DataLoader(
    dataset=raw_dataset,
    batch_sampler=batch_sampler,
    num_workers=num_workers,
    collate_fn=batchify, # Important for batching custom data types
)

# 4. Define Draw Frame Callbacks for visualization overlays
draw_frame_callbacks = [
    ActionDrawFrameCallback(), 
    MetaInfoDrawFrameCallback(), 
    SegmentationDrawFrameCallback() # If segmentation is visualized
]

# 5. Visualize
visualize_dataloader(
    dataloader, 
    draw_frame_callbacks=draw_frame_callbacks,
    num_samples=num_samples_to_viz, 
    save_fps=save_fps_viz,
    # output_dir="./my_raw_videos/" # Optional
)
```

This setup will produce videos where each video represents a batch of sequences. The `ActionDrawFrameCallback`, `MetaInfoDrawFrameCallback`, etc., will draw relevant information on the frames.

Here is an example video (conceptual, replace with actual output if available):
```{youtube} JvlFptYjOm0
```

## Visualize Batches with Special Events (Using `EventDataset`)

To visualize data segments centered around specific in-game events, use `EventDataset`. This dataset loads windows of multi-modal data triggered by events matching a regular expression.

```python
import argparse
from torch.utils.data import DataLoader

from minestudio.data import EventDataset
from minestudio.data.minecraft.utils import batchify, visualize_dataloader
from minestudio.data.minecraft.callbacks import (
    ImageKernelCallback, 
    ActionKernelCallback, 
    MetaInfoKernelCallback, 
    SegmentationKernelCallback, 
    ActionDrawFrameCallback, 
    MetaInfoDrawFrameCallback, 
    SegmentationDrawFrameCallback
)

# Configuration (similar to args in test_viz_event.py)
dataset_dirs = ['/nfs-shared-2/data/contractors/dataset_6xx'] # Replace with your dataset path(s)
win_len = 64
skip_frame = 1 # skip_frame is also applicable to EventDataset if needed by callbacks or post-processing
frame_width = 224
frame_height = 224
enable_video_aug = False
batch_size = 4 # Number of event windows per batch
num_workers = 2
event_regex = 'minecraft.mine_block:.*diamond.*' # Example: mining diamond ore
min_nearby = 64 # Optional: min ticks/frames between same events
max_within = 1000 # Optional: max instances for an event type
num_samples_to_viz = 3
save_fps_viz = 20

# 1. Define Modal Kernel Callbacks
modal_kernel_callbacks = [
    ImageKernelCallback(
        frame_width=frame_width, 
        frame_height=frame_height, 
        enable_video_aug=enable_video_aug,
    ), 
    ActionKernelCallback(),
    MetaInfoKernelCallback(),
    SegmentationKernelCallback(
        frame_width=frame_width, 
        frame_height=frame_height, 
    )
]

# 2. Create EventDataset
event_dataset = EventDataset(
    dataset_dirs=dataset_dirs, 
    modal_kernel_callbacks=modal_kernel_callbacks,
    win_len=win_len, 
    # skip_frame=skip_frame, # Pass if EventDataset or its components use it
    event_regex=event_regex,
    min_nearby=min_nearby,
    max_within=max_within,
    # bias can also be set here if needed, e.g., bias=-32 for centering a window of 64
)

# 3. Create DataLoader
dataloader = DataLoader(
    dataset=event_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=batchify, # Important for batching
    shuffle=True # Usually shuffle event samples
)

# 4. Define Draw Frame Callbacks
draw_frame_callbacks = [
    ActionDrawFrameCallback(), 
    MetaInfoDrawFrameCallback(), 
    SegmentationDrawFrameCallback()
]

# 5. Visualize
visualize_dataloader(
    dataloader, 
    draw_frame_callbacks=draw_frame_callbacks,
    num_samples=num_samples_to_viz, 
    save_fps=save_fps_viz,
    # output_dir="./my_event_videos/" # Optional
)
```

This will generate videos of data windows centered around the specified events.

Here is an example video (conceptual, replace with actual output if available):
```{youtube} 9YU3y0ZWh8Y
```
