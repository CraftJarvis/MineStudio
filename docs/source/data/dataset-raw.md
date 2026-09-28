# Raw Dataset

The Raw Dataset refers to a simple way of reading the original data, which stores the raw trajectory segments in chronological order. 
```{hint}
Users can choose to read random segments from it or opt to read segments continuously in chronological order. 
```

## Basic Information

Here are the primary arguments of the `RawDataset` class:

| Arguments                | Description                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `dataset_dirs`           | A list of strings, where each string is a path to a directory containing the dataset.                      |
| `modal_kernel_callbacks` | A list of `ModalKernelCallback` instances (e.g., `ImageKernelCallback()`, `ActionKernelCallback()`). This determines which data modalities are loaded and how they are processed. Each callback corresponds to a specific modality. |
| `modal_kernel_config`    | Optional. A dictionary to configure modal kernels if `modal_kernel_callbacks` are provided as strings (advanced usage). |
| `seed`                   | An integer for the random seed used for shuffling episodes. Defaults to `0`.                               |
| `win_len`                | An integer representing the window length (number of frames) for each item/segment. Defaults to `1`.         |
| `skip_frame`             | An integer indicating the number of frames to skip when building an item/segment. Defaults to `1`.           |
| `split`                  | A string specifying the dataset split, either `'train'` or `'val'`. Defaults to `'train'`.                 |
| `split_ratio`            | A float representing the ratio for splitting the dataset into training and validation sets. Defaults to `0.8`. |
| `verbose`                | A boolean. If `True`, prints verbose information during dataset initialization. Defaults to `True`.        |
| `shuffle_episodes`       | A boolean. If `True`, shuffles the episodes before splitting and item creation. Defaults to `False`.       |

Modalities like video (image), actions, metadata, and segmentation are no longer enabled by simple boolean flags. Instead, you provide a list of specific callback objects (e.g., `ImageKernelCallback`, `ActionKernelCallback`, `MetaInfoKernelCallback`, `SegmentationKernelCallback`) to the `modal_kernel_callbacks` argument. Parameters like `frame_width` and `frame_height` are configured within the respective callbacks (e.g., `ImageKernelCallback(frame_width=224, frame_height=224)`).

## Loading Segment-level Data

When the user does not have a need to process long trajectories, segments from the same trajectory are independent and can be read randomly. This reading method is suitable for some simple tasks, such as training a policy that can perform short-range tasks, like GROOT-1. At this point, the user only needs to wrap `RawDataset` with PyTorch's built-in dataloader to achieve data reading.

Here is an example of how to load the segment-level data:

```python
from torch.utils.data import DataLoader
from minestudio.data import RawDataset
from minestudio.data.minecraft.callbacks import (
    ImageKernelCallback,
    ActionKernelCallback,
    MetaInfoKernelCallback,
    SegmentationKernelCallback
)
from minestudio.data.minecraft.utils import batchify # Ensure this utility is appropriate for your batching needs

# Define the modal kernel callbacks for the modalities you want to load
# These instances configure how each modality's data is handled.
modal_callbacks = [
    ImageKernelCallback(frame_width=224, frame_height=224), # For image data
    ActionKernelCallback(),                                 # For action data
    MetaInfoKernelCallback(),                               # For metadata
    SegmentationKernelCallback(frame_width=224, frame_height=224) # For segmentation data
]

dataset = RawDataset(
    dataset_dirs=[
        '/nfs-shared-2/data/contractors/dataset_6xx', # Example dataset directory
        # Add more dataset directories if needed
    ],
    modal_kernel_callbacks=modal_callbacks,
    win_len=128,
    skip_frame=1,
    split='train',
    split_ratio=0.8,
    verbose=True,
    shuffle_episodes=True, # Example: shuffle episodes
    seed=42                # Example: set a seed for reproducibility
)

# Use PyTorch's DataLoader for batching, collate_fn might be needed depending on item structure
# 'batchify' is used here as in the original example; ensure it's compatible.
loader = DataLoader(dataset, batch_size=4, collate_fn=batchify)

for item_batch in loader:
    print(f"Batch keys: {item_batch.keys()}")
    if 'image' in item_batch:
        print(f"Image batch shape: {item_batch['image'].shape}")
    if 'meta_info' in item_batch and isinstance(item_batch['meta_info'], dict): # meta_info is often a dict of tensors
        print(f"Meta info keys: {item_batch['meta_info'].keys()}")
    # Print other relevant information about the batch
    break
```

Now, you can see output similar to the following (the exact keys and shapes will depend on the callbacks used and their configurations):

```
Batch keys: dict_keys(['image', 'image_mask', 'action_mask', 'env_action', 'agent_action', 'meta_info', 'meta_info_mask', 'segmentation', 'segmentation_mask', 'mask', 'text', 'timestamp', 'episode', 'progress'])
Image batch shape: torch.Size([4, 128, 224, 224, 3])
Meta info keys: dict_keys(['yaw', 'pitch', 'xpos', 'ypos', 'zpos', 'hotbar', 'inventory', 'isGuiOpen', 'isGuiInventory', 'delta_yaw', 'delta_pitch', 'events', 'cursor_x', 'cursor_y'])
```

## Loading Episode-level Data

When you need to process long trajectories where segments from the same episode are related and must be read in order (episode continuity), the `RawDataModule` provides a convenient way to achieve this. This approach is suitable for tasks requiring long-range dependencies, such as training certain types of policies (e.g., VPT).

By setting `episode_continuous_batch=True` when creating the `RawDataModule`, it internally uses a specialized sampler (like `MineDistributedBatchSampler`) to ensure that each slot in a batch maintains the chronological order of frames within an episode. When an episode runs out of segments, that slot in the batch is then filled with a new episode.

Here is an example of how to load episode-level data using `RawDataModule`:

```python
from minestudio.data import RawDataModule # Make sure this import path is correct for your project
from minestudio.data.minecraft.callbacks import (
    ImageKernelCallback,
    ActionKernelCallback,
    MetaInfoKernelCallback,
    SegmentationKernelCallback
)
# Ensure other necessary utilities like a collate_fn (e.g., batchify) are available if needed by DataLoader

# 1. Define the modal kernel callbacks for the modalities you want to load
modal_callbacks = [
    ImageKernelCallback(frame_width=224, frame_height=224),
    ActionKernelCallback(), 
    MetaInfoKernelCallback(),
    SegmentationKernelCallback(frame_width=224, frame_height=224),
]

# 2. Configure and instantiate RawDataModule
data_module = RawDataModule(
    data_params=dict(
        dataset_dirs=[
            '/nfs-shared-2/data/contractors/dataset_10xx', # Replace with your dataset path(s)
        ],
        modal_kernel_callbacks=modal_callbacks,
        win_len=128,         # Window length for each item
        skip_frame=1,        # Frames to skip when building items
        split_ratio=0.9,     # Train/val split ratio
        shuffle_episodes=True, # Shuffle episodes before splitting
        seed=42,             # Seed for reproducibility
    ),
    batch_size=4,          # Number of trajectories processed in parallel per batch
    num_workers=8,         # Number of worker processes for data loading
    episode_continuous_batch=True, # Crucial for episode-level data continuity
)

# 3. Setup the DataModule (this prepares train_dataset, val_dataset, etc.)
data_module.setup()

# 4. Get the DataLoader
# For episode-level data, this loader will yield batches maintaining episode continuity.
train_loader = data_module.train_dataloader()

# 5. Iterate through the DataLoader
print("Iterating through episode-continuous batches (episode_name progress):")
for idx, batch in enumerate(train_loader):
    # The 'episode' and 'progress' keys in the batch allow tracking of continuous trajectories.
    # Their exact format and availability depend on the dataset and batching implementation.
    if 'episode' in batch and 'progress' in batch:
        batch_info_parts = []
        for i in range(len(batch['episode'])):
            ep_name = batch['episode'][i]
            ep_progress = batch['progress'][i]
            batch_info_parts.append(f"{str(ep_name)[-30:]} {str(ep_progress)}") # Show last 30 chars of name
        print("\t".join(batch_info_parts))
    else:
        # Fallback if 'episode' or 'progress' keys are not directly available in the batch
        print(f"Batch {idx+1} loaded. Keys: {batch.keys()}")

    if idx >= 5:  # Limit the number of printed batches for brevity in documentation
        break
```

Now, you can see output similar to the following, where each column represents a slot in the batch processing an episode continuously:
```
Iterating through episode-continuous batches (episode_name progress):
a-92de05e1a4b2-20220421-052900 0/4      r-f153ac423f61-20220419-123621 0/15     a-24f3e4f55656-20220417-160454 0/151    a-48bf00edae01-20220421-043237 0/161
a-92de05e1a4b2-20220421-052900 1/4      r-f153ac423f61-20220419-123621 1/15     a-24f3e4f55656-20220417-160454 1/151    a-48bf00edae01-20220421-043237 1/161
a-92de05e1a4b2-20220421-052900 2/4      r-f153ac423f61-20220419-123621 2/15     a-24f3e4f55656-20220417-160454 2/151    a-48bf00edae01-20220421-043237 2/161
a-92de05e1a4b2-20220421-052900 3/4      r-f153ac423f61-20220419-123621 3/15     a-24f3e4f55656-20220417-160454 3/151    a-48bf00edae01-20220421-043237 3/161
r-33cef7a39444-20220419-160613 0/139    r-f153ac423f61-20220419-123621 4/15     a-24f3e4f55656-20220417-160454 4/151    a-48bf00edae01-20220421-043237 4/161
r-33cef7a39444-20220419-160613 1/139    r-f153ac423f61-20220419-123621 5/15     a-24f3e4f55656-20220417-160454 5/151    a-48bf00edae01-20220421-043237 5/161
```

```{note}
The `RawDataModule` (when `episode_continuous_batch=True`) internally uses a sampler like `MineDistributedBatchSampler`. This sampler ensures that each batch slot maintains the order of the trajectory. Only when a trajectory runs out of segments will the slot be filled with a new trajectory.
```

```{note}
When using a distributed training strategy, the underlying `MineDistributedBatchSampler` (or a similar one used by `RawDataModule`) will automatically divide the dataset among the GPUs. Most episodes will belong to only one GPU's part. If an episode is split across parts, each part is typically treated as a new, shorter episode for loading purposes.
```

If you need more fine-grained control or are not using the `RawDataModule`, you might interact with `MineDistributedBatchSampler` directly. Here are its arguments:

| Arguments | Description |
| --- | --- |
| dataset | the dataset to sample from |
| batch_size | how many samples per batch to load |
| num_replicas | the number of processes participating in the training; lightning will set this for you |
| rank | the rank of the current process within num_replicas; lightning will set this for you |
| shuffle | must be `false`, you can do shuffle operation in the `RawDataset` |
| drop_last | must be `true` |

## Using Lightning Fabric for Distributed Data Loading

PyTorch Lightning Fabric can simplify setting up distributed training, including data loading. When using `RawDataModule` with `episode_continuous_batch=True`, Fabric can correctly handle the distributed setup.

Here is an example demonstrating how to use `RawDataModule` with Lightning Fabric for distributed data loading with episode continuity:

```python
import lightning as L
from tqdm import tqdm # Optional: for progress bars

from minestudio.data import RawDataModule
from minestudio.data.minecraft.callbacks import (
    ImageKernelCallback, ActionKernelCallback, SegmentationKernelCallback # Add other callbacks as needed
)

# This flag should be True for episode-level continuity
continuous_batch = True

# 1. Initialize Lightning Fabric
# Adjust accelerator, devices, and strategy as per your setup
fabric = L.Fabric(accelerator="cuda", devices=2, strategy="ddp") 
fabric.launch() # Important for DDP initialization

# 2. Define Modal Kernel Callbacks
modal_callbacks = [
    ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False),
    ActionKernelCallback(),
    SegmentationKernelCallback(frame_width=224, frame_height=224),
]

# 3. Configure and instantiate RawDataModule
data_module = RawDataModule(
    data_params=dict(
        dataset_dirs=[
            '/nfs-shared-2/data/contractors-new/dataset_6xx', # Replace with your actual dataset paths
            '/nfs-shared-2/data/contractors-new/dataset_7xx',
            # Add more dataset directories if needed
        ],
        modal_kernel_callbacks=modal_callbacks,
        win_len=128,
        split_ratio=0.9,
        shuffle_episodes=True,
        seed=42, # Optional: for reproducibility
    ),
    batch_size=4, # This will be the batch size per process
    num_workers=2,
    prefetch_factor=4, # Optional: for performance tuning
    episode_continuous_batch=continuous_batch, 
)

# 4. Setup the DataModule
data_module.setup() # This prepares the datasets

# 5. Get the DataLoader
train_loader = data_module.train_dataloader()

# 6. Setup DataLoader with Fabric
# When episode_continuous_batch is True, RawDataModule's internal sampler handles distribution.
# So, use_distributed_sampler should be False for Fabric, as the sampler is already DDP-aware.
train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=not continuous_batch)

# 7. Iterate through the DataLoader
rank = fabric.local_rank # Get the rank of the current process
print(f"Rank {rank} starting iteration (episode_name progress)...")
for idx, batch in enumerate(tqdm(train_loader, disable=(rank != 0))): # tqdm only on rank 0
    if idx > 5: # Limit printed batches for brevity
        break
    
    # Example: Print episode and progress from the batch
    if 'episode' in batch and 'progress' in batch:
        batch_info_parts = []
        for i in range(len(batch['episode'])):
            ep_name = batch['episode'][i]
            ep_progress = batch['progress'][i]
            batch_info_parts.append(f"{str(ep_name)[-30:]} {str(ep_progress)}")
        print(
            f"Rank {rank} - Batch {idx+1}: \t" + "\t".join(batch_info_parts)
        )
    else:
        print(f"Rank {rank} - Batch {idx+1} keys: {batch.keys()}")

```
Here is an example of the expected output (the exact episode names and progress will vary):
```
Rank 0 starting iteration (episode_name progress)...
Rank 0 - Batch 1:   ..._episode_X_rank0 0/100   ..._episode_Y_rank0 0/120
Rank 1 starting iteration (episode_name progress)...
Rank 1 - Batch 1:   ..._episode_A_rank1 0/110   ..._episode_B_rank1 0/90
Rank 0 - Batch 2:   ..._episode_X_rank0 1/100   ..._episode_Y_rank0 1/120
Rank 1 - Batch 2:   ..._episode_A_rank1 1/110   ..._episode_B_rank1 1/90
...
```

```{note}
As seen in the output, each distributed process (rank) receives its own batches, and within those batches, the episode continuity is maintained for each slot. The key is that `RawDataModule` with `episode_continuous_batch=True` provides a DataLoader whose sampler is already aware of distributed training. Therefore, `fabric.setup_dataloaders` should be called with `use_distributed_sampler=False` (or `use_distributed_sampler=not continuous_batch` as in the example) to avoid conflicts.
```