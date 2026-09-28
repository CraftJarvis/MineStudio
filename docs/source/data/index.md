# Data

We design a trajectory structure for storing Minecraft data. Based on this data structure, users are able to store and retrieve arbitray trajectory segment in an efficient way. 

```{toctree}
:caption: MineStudio Data

dataset-raw
dataset-event
visualization
convertion
callbacks
```

## Quick Start
````{include} quick-data.md
````

## Data Structure

We classify and save the data according to its corresponding modality, with each modality's data being a sequence over time. Sequences from different modalities can be aligned in chronological order. For example, the "action" modality data stores the mouse and keyboard actions taken at each time step of the trajectory; the "video" modality data stores the observations returned by the environment at each time step of the trajectory. 

```{note}
The data of different modalities is stored independently. The benefits are: (1) Users can selectively read data from different modalities according to their requirements; (2) Users are easily able to add new modalities to the dataset without affecting the existing data. 
```

For the sequence data of each modality, we store it in segments, with each segment having a fixed length (e.g., 32), which facilitates the reading and storage of the data. 

```{note}
For video data, the efficiency of random access is usually low because decoding is required during the reading process. An extreme case would be to save it as individual images, which would allow for high read efficiency but take up a large amount of storage space. 

We adopt a compromise solution by saving the video data in video segments, which allows for relatively high read efficiency while not occupying too much storage space. When user wants to read a sequence of continuous frames, we only need to retrieve the corresponding segments and decode them. 
```

```{image} ./read_video_fig.png
:width: 80%
```

````{dropdown} <i class="fa-solid fa-lightbulb" height="35px" width="20px"></i> Learn more about the details

Segmented sequence data is stored in individual [lmdb](https://lmdb.readthedocs.io/en/release/) files, each of which contains the following metadata: 
```python
{
    "__num_episodes__": int,     # the total number of episodes in this lmdb file
    "__num_total_frames__": int, # the total number of frames in this lmdb file
    "__chunk_size__": int,       # the length of each segment (e.g. 32)
    "__chunk_infos__": dict      # save the information of the episode part in this lmdb file, e.g. the start and end index, episode name. 
}
```

Once you know the episode name and which segment you want to read, you can identify the corresponding segment bytes in the lmdb file and decode it to get the data. 

```python
with lmdb_handler.begin() as txn:
    key = str((episode_idx, chunk_id)).encode()
    chunk_bytes = txn.get(key)
```

```{hint}
In fact, you don't need to worry about these low-level details, as we have packaged these operations for you. You just need to call the corresponding API. The class primarily responsible for managing these details for a single data modality is `minestudio.data.minecraft.core.ModalKernel`. For managing multiple modalities, `minestudio.data.minecraft.core.KernelManager` is used, which internally utilizes `ModalKernel` instances.
```

With `ModalKernel`, you can perform these operations on the data:
- Get the list of episodes (trajectories):
    ```python
    # Assuming 'modal_kernel' is an instance of ModalKernel
    episode_list = modal_kernel.get_episode_list()
    ```

- Get the total number of frames for specified episodes:
    ```python
    # Assuming 'modal_kernel' is an instance of ModalKernel
    total_frames = modal_kernel.get_num_frames([
        "episode_1",
        "episode_2",
        "episode_3"
    ])
    ```

- Read a sequence of frames from an episode:
    ```python
    # Assuming 'modal_kernel' is an instance of ModalKernel
    # and it has been initialized with an appropriate ModalKernelCallback
    data_dict = modal_kernel.read_frames(
        eps="episode_1",
        start=11,      # Starting frame index
        win_len=33,    # Window length (number of frames to read)
        skip_frame=1   # Number of frames to skip between reads (1 means no skip)
    )
    # 'data_dict' will contain the processed frames and potentially other info like a mask,
    # depending on the ModalKernelCallback.
    ```

    ```{note}
    The processing of data, such as merging chunks, extracting specific information, and padding, is handled by a `ModalKernelCallback` instance that is provided when the `ModalKernel` is created. This callback is specific to the data modality (e.g., video, actions).
    ```

````

### Built-in Modalities

We provide the following built-in modalities for users to store data:

| Modality       | Description                                         | Data Format   |
|----------------|-----------------------------------------------------|---------------|
| image          | Visual observations from the environment (frames)   | `np.ndarray`  |
| action         | Player/agent's mouse and keyboard inputs            | `Dict`        |
| meta_info      | Auxiliary information about the game state/episode  | `Dict`        |
| segmentation   | Object segmentation masks for the visual frames     | `Dict`        |

````{admonition} Video and Segmentation Visualization
:class: dropdown admonition-youtube
<!-- 
An video example generated by our tool to show video and the corresponding segmentation sequences.  -->

```{youtube} QYBUxus3esI
```
````
