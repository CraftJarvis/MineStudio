Here is a minimal example to show how we load trajectories from the dataset. 

```python
from minestudio.data import RawDataset
from minestudio.data.minecraft.callbacks import ImageKernelCallback, ActionKernelCallback

dataset = RawDataset(
    dataset_dirs=[
        '/nfs-shared-2/data/contractors/dataset_6xx', 
        '/nfs-shared-2/data/contractors/dataset_7xx', 
        '/nfs-shared-2/data/contractors/dataset_8xx', 
        '/nfs-shared-2/data/contractors/dataset_9xx', 
        '/nfs-shared-2/data/contractors/dataset_10xx', 
    ],
    modal_kernel_callbacks=[
        ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False), 
        ActionKernelCallback(enable_prev_action=True, win_bias=1, read_bias=-1),
    ],
    win_len=128, 
    split_ratio=0.9,
    shuffle_episodes=True,
)
item = dataset[0]
print(item.keys())
```

You may see the output like this: 
```
[23:09:38] [Kernel] Modal image load 15738 episodes.
[23:09:38] [Kernel] Modal action load 15823 episodes.
[23:09:38] [Kernel] episodes: 15655, frames: 160495936.
[Raw Dataset] Shuffling episodes with seed 0. 
dict_keys(['image', 'image_mask', 'action_mask', 'env_action', 'env_prev_action', 'agent_action', 'agent_prev_action', 'mask', 'text', 'timestamp', 'episode', 'progress'])
```

```{hint}
Please note that the `dataset_dirs` parameter here is a list that can contain multiple dataset directories. In this example, we have loaded five dataset directories. 

If an element in the list is one of `6xx`, `7xx`, `8xx`, `9xx`, or `10xx`, the program will automatically download it from [Hugging Face](https://huggingface.co/CraftJarvis), so please ensure your network connection is stable and you have enough storage space. 
If an element in the list is a directory like `/nfs-shared/data/contractors/dataset_6xx`, the program will load data directly from that directory. 

**We strongly recommend users to manually download the datasets and place them in a local directory, such as `/nfs-shared-2/data/contractors/dataset_6xx`, to avoid downloading issues.**
```


```{button-ref}  ./dataset-raw
:color: primary
:outline:
:expand:

Learn more about Raw Dataset
```

Alternatively, you can also load trajectories that have specific events, for example, loading all trajectories that contain the ``kill entity`` event. 

```python
from minestudio.data import EventDataset
from minestudio.data.minecraft.callbacks import ImageKernelCallback, ActionKernelCallback

dataset = EventDataset(
    dataset_dirs=[
        '/nfs-shared-2/data/contractors/dataset_6xx', 
        '/nfs-shared-2/data/contractors/dataset_7xx', 
        '/nfs-shared-2/data/contractors/dataset_8xx', 
        '/nfs-shared-2/data/contractors/dataset_9xx', 
        '/nfs-shared-2/data/contractors/dataset_10xx', 
    ], 
    modal_kernel_callbacks=[
        ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False), 
        ActionKernelCallback(),
    ],
    win_len=128, 
    split_ratio=0.9, 
    event_regex='minecraft.kill_entity:.*', 
    min_nearby=64,
    max_within=1000,
)
print("length of dataset: ", len(dataset))
item = dataset[0]
print(item.keys())
```

You may see the output like this: 
```
[23:16:07] [Event Kernel Manager] Number of loaded events: 61
[23:16:07] [Kernel] Modal image load 15738 episodes.
[23:16:07] [Kernel] Modal action load 15823 episodes.
[23:16:07] [Kernel] episodes: 15655, frames: 160495936.
[23:16:07] [Event Dataset] Regex: minecraft.kill_entity:.*, Number of events: 61, number of items: 16835. 
length of dataset: 16835
dict_keys(['image', 'env_action', 'agent_action', 'mask', 'text', 'episode', 'timestamp'])
```

```{button-ref}  ./dataset-event
:color: primary
:outline:
:expand:

Learn more about Event Dataset
```

We also provide a dataloader wrapper to make it easier to use the dataset. 

```python
from minestudio.data import RawDataModule
from minestudio.data.minecraft.callbacks import (
    ImageKernelCallback, ActionKernelCallback
)

data_module = RawDataModule(
    data_params=dict(
        dataset_dirs=[
            '/nfs-shared-2/data/contractors/dataset_6xx', 
            '/nfs-shared-2/data/contractors/dataset_7xx', 
            '/nfs-shared-2/data/contractors/dataset_8xx', 
            '/nfs-shared-2/data/contractors/dataset_9xx', 
            '/nfs-shared-2/data/contractors/dataset_10xx', 
        ],
        modal_kernel_callbacks=[
            ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False), 
            ActionKernelCallback(enable_prev_action=True, win_bias=1, read_bias=-1),
        ],
        win_len=128, 
        split_ratio=0.9,
        shuffle_episodes=True,
    ),
    batch_size=3,
    num_workers=8,
    prefetch_factor=None,
    episode_continuous_batch=True,
)
data_module.setup()
loader = data_module.train_dataloader()
for idx, batch in enumerate(loader):
    print(
        "\t".join(
            [f"{a} {b}" for a, b in zip(batch['episode'], batch['progress'])]
        )
    )
    if idx > 10:
        break
```

You may see the output like this: 
```
[23:21:03] [Kernel] Modal image load 15738 episodes.
[23:21:03] [Kernel] Modal action load 15823 episodes.
[23:21:03] [Kernel] episodes: 15655, frames: 160495936.
[Raw Dataset] Shuffling episodes with seed 0. 
[23:21:03] [Kernel] Modal image load 15738 episodes.
[23:21:03] [Kernel] Modal action load 15823 episodes.
[23:21:03] [Kernel] episodes: 15655, frames: 160495936.
[Raw Dataset] Shuffling episodes with seed 0. 
thirsty-lavender-koala-7552d1728d4d-20220411-092042 0/75        Player63-f153ac423f61-20210723-162533 0/8       wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 0/88
thirsty-lavender-koala-7552d1728d4d-20220411-092042 1/75        Player63-f153ac423f61-20210723-162533 1/8       wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 1/88
thirsty-lavender-koala-7552d1728d4d-20220411-092042 2/75        Player63-f153ac423f61-20210723-162533 2/8       wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 2/88
thirsty-lavender-koala-7552d1728d4d-20220411-092042 3/75        Player63-f153ac423f61-20210723-162533 3/8       wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 3/88
thirsty-lavender-koala-7552d1728d4d-20220411-092042 4/75        Player63-f153ac423f61-20210723-162533 4/8       wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 4/88
thirsty-lavender-koala-7552d1728d4d-20220411-092042 5/75        Player63-f153ac423f61-20210723-162533 5/8       wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 5/88
thirsty-lavender-koala-7552d1728d4d-20220411-092042 6/75        Player63-f153ac423f61-20210723-162533 6/8       wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 6/88
thirsty-lavender-koala-7552d1728d4d-20220411-092042 7/75        Player63-f153ac423f61-20210723-162533 7/8       wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 7/88
thirsty-lavender-koala-7552d1728d4d-20220411-092042 8/75        Player985-f153ac423f61-20210914-114117 0/23     wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 8/88
thirsty-lavender-koala-7552d1728d4d-20220411-092042 9/75        Player985-f153ac423f61-20210914-114117 1/23     wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 9/88
thirsty-lavender-koala-7552d1728d4d-20220411-092042 10/75       Player985-f153ac423f61-20210914-114117 2/23     wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 10/88
thirsty-lavender-koala-7552d1728d4d-20220411-092042 11/75       Player985-f153ac423f61-20210914-114117 3/23     wiggy-aquamarine-tapir-c09d137a3840-20220318-024035 11/88
```