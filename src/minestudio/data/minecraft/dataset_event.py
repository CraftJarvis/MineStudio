'''
Date: 2024-11-10 10:26:52
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-05-27 15:24:46
FilePath: /MineStudio/minestudio/data/minecraft/dataset_event.py
'''
import io
import re
import os
import lmdb
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as L

from rich.console import Console
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Sequence, Mapping, Any, Optional, Literal

from minestudio.data.minecraft.utils import batchify
from minestudio.data.minecraft.core import KernelManager
from minestudio.data.minecraft.callbacks import ModalKernelCallback
from minestudio.utils.register import Registers

class EventKernel:
    """
    Manages and provides access to event data stored in an LMDB database.

    This class reads event information and specific event items from an LMDB
    database. It allows filtering events based on a regular expression and
    can further filter events by minimum time separation (min_nearby) or
    a maximum number of events to consider within a larger set (max_within).
    It also handles a codebook for mapping episode names if provided.

    :param event_path: Path to the directory containing the LMDB event database.
    :type event_path: Union[str, Path]
    :param event_regex: Regular expression to filter event names.
    :type event_regex: str
    :param min_nearby: Optional minimum time (in game ticks or similar units)
                       between two occurrences of the same event in the same episode
                       for an event instance to be included. Defaults to None.
    :type min_nearby: Optional[int]
    :param max_within: Optional maximum number of event instances to consider for each
                       event type after other filtering. Defaults to None.
    :type max_within: Optional[int]
    """
    
    def __init__(self, event_path: Union[str, Path], event_regex: str, min_nearby: Optional[int] = None, max_within: Optional[int] = None) -> None:
        if isinstance(event_path, str):
            event_path = Path(event_path)
        assert event_path.is_dir(), f"Event lmdb file {event_path} does not exist. "
        self.lmdb_stream = lmdb.open(str(event_path), max_readers=128, readonly=True, lock=False)

        with self.lmdb_stream.begin(write=False) as txn:
            __event_info__ = pickle.loads(txn.get(b'__event_info__'))
            # check if codebook exists
            __codebook_bytes__ = txn.get(b'__codebook__', None)
            if __codebook_bytes__ is None:
                self.__codebook__ = None
            else:
                self.__codebook__ = {v: k for k, v in pickle.loads(__codebook_bytes__).items()}
            self.event_info = {}
            for event, value in __event_info__.items():
                if re.match(event_regex, event):
                    self.event_info[event] = value
        
        self.event_list = sorted(list(self.event_info.keys()))
        # if min_nearby is not None or max_within is not None:
        self.filter_out(min_nearby, max_within)
    
    def filter_out(self, min_nearby: Optional[int] = None, max_within: Optional[int] = None):
        """
        Filters the events based on proximity and count constraints.

        This method refines the list of events to be used. It ensures that for any given
        episode and event type, consecutive occurrences are separated by at least `min_nearby`
        time units. It also limits the total number of occurrences for each event type to
        `max_within` if specified.

        :param min_nearby: Optional minimum time between consecutive events of the same type
                           within an episode. If an event occurs too close to the previous one,
                           it's filtered out. Defaults to None.
        :type min_nearby: Optional[int]
        :param max_within: Optional maximum number of instances to keep for each event type
                           after applying the `min_nearby` filter. Defaults to None.
        :type max_within: Optional[int]
        """
        episode_event_last = {}
        remaining_events = {}
        for event in self.event_list:
            num_events = self.get_event_size(event)
            remaining_events[event] = []
            for i in range(num_events):
                episode, event_time, value = self.get_event_item(event, i)
                if event_time < 128: # remove dirty events
                    continue
                episode_event_key = f"{episode}:{event}"
                if episode_event_key not in episode_event_last:
                    episode_event_last[episode_event_key] = -100000

                if min_nearby is not None \
                    and event_time - episode_event_last[episode_event_key] <= min_nearby:
                    continue
                
                if max_within is not None \
                    and len(remaining_events[event]) >= max_within:
                    break
                
                episode_event_last[episode_event_key] = event_time
                remaining_events[event].append(i)
            self.event_info[event]['__num_items__'] = len(remaining_events[event])
        self.remaining_events = remaining_events
    
    def get_event_list(self) -> List[str]:
        """
        Returns the sorted list of unique event names that match the `event_regex`
        and are present in the loaded event data.

        :returns: A list of event names.
        :rtype: List[str]
        """
        return self.event_list
    
    def get_event_size(self, event: str) -> int:
        """
        Returns the number of occurrences for a specific event type after filtering.

        :param event: The name of the event.
        :type event: str
        :returns: The count of occurrences for the given event. Returns 0 if the event
                  is not found or has no occurrences after filtering.
        :rtype: int
        """
        if event not in self.event_info:
            return 0
        return self.event_info[event]['__num_items__']

    def get_event_item(self, event: str, item_idx: int) -> Tuple[str, int, int]:
        """
        Retrieves a specific event item by its type and index.

        The method first remaps the `item_idx` if filtered `remaining_events` exist.
        It then fetches the raw event item (episode, event_time, value) from the LMDB
        database. If a codebook is present, it maps the episode name.

        :param event: The name of the event type (e.g., "minecraft.kill_entity:zombie").
        :type event: str
        :param item_idx: The index of the desired event item within its type, after filtering.
        :type item_idx: int
        :raises AssertionError: If `item_idx` is out of the valid range for the given event.
        :returns: A tuple containing:
                  - episode (str): The name or ID of the episode where the event occurred.
                  - event_time (int): The timestamp (e.g., game tick) of the event.
                  - value (int): A value associated with the event (semantics depend on event type).
        :rtype: Tuple[str, int, int]
        """
        assert item_idx < self.get_event_size(event), f"Item index {item_idx} out of range. "
        if hasattr(self, 'remaining_events'):
            item_idx = self.remaining_events[event][item_idx] # remap the index
        key = str((event, item_idx))
        with self.lmdb_stream.begin(write=False) as txn:
            item = pickle.loads(txn.get(key.encode()))
        episode, event_time, value = item
        if self.__codebook__ is not None:
            episode = self.__codebook__[episode]
        return episode, event_time, value

class EventKernelManager:
    """
    Manages multiple EventKernel instances to provide a unified view of event data.

    This class aggregates event data from several EventKernel objects, which might
    correspond to different event LMDB databases or different partitions of data.
    It allows querying the total size of an event type across all kernels and
    retrieving specific event items by a global index.

    :param event_path: A list of paths to directories, each containing an LMDB event database
                       to be managed by an EventKernel.
    :type event_path: List[Union[str, Path]]
    :param event_regex: Regular expression used by each underlying EventKernel to filter event names.
    :type event_regex: str
    :param verbose: If True, logs information about the loaded events, defaults to True.
    :type verbose: bool, optional
    :param \*\*kwargs: Additional keyword arguments passed to the constructor of each EventKernel
                     (e.g., `min_nearby`, `max_within`).
    """
    
    def __init__(self, event_path: List[Union[str, Path]], event_regex: str, verbose: bool = True, **kwargs) -> None:
        self.verbose = verbose
        self.event_kernels = [EventKernel(event, event_regex, **kwargs) for event in event_path]
        event_set = set()
        for kernel in self.event_kernels:
            event_set.update(kernel.get_event_list())
        self.event_list = sorted(list(event_set))
        if verbose:
            Console().log(f"[Event Kernel Manager] Number of loaded events: {len(self.event_list)}")
    
    def get_event_list(self) -> List[str]:
        """
        Returns the sorted list of unique event names aggregated from all managed EventKernel instances.

        :returns: A list of unique, sorted event names.
        :rtype: List[str]
        """
        return self.event_list
    
    def get_event_size(self, event: str) -> int:
        """
        Calculates the total number of occurrences for a specific event type across all managed kernels.

        :param event: The name of the event.
        :type event: str
        :returns: The total count of the event. Returns 0 if the event is not found in any kernel
                  or has no occurrences after filtering in the underlying kernels.
        :rtype: int
        """
        if event not in self.event_list:
            return 0
        return sum([kernel.get_event_size(event) for kernel in self.event_kernels])
    
    def get_event_item(self, event: str, item_idx: int) -> Tuple[str, int, int]:
        """
        Retrieves a specific event item by its type and a global index across all managed kernels.

        It iterates through the managed EventKernel instances, subtracting the size of each kernel's
        event pool from `item_idx` until the index falls within the range of the current kernel.
        Then, it calls the `get_event_item` method of that specific kernel.

        :param event: The name of the event type.
        :type event: str
        :param item_idx: The global index of the desired event item across all kernels.
        :type item_idx: int
        :raises ValueError: If `item_idx` is out of the valid range for the given event
                           across all managed kernels.
        :returns: A tuple containing:
                  - episode (str): The name or ID of the episode.
                  - event_time (int): The timestamp of the event.
                  - value (int): A value associated with the event.
        :rtype: Tuple[str, int, int]
        """
        for kernel in self.event_kernels:
            size = kernel.get_event_size(event)
            if item_idx < size:
                return kernel.get_event_item(event, item_idx)
            item_idx -= size
        raise ValueError(f"Item index {item_idx} out of range. ")

class EventDataset(Dataset):
    """
    A PyTorch Dataset for loading sequences of data centered around specific game events.

    This dataset uses an `EventKernelManager` to identify occurrences of specified events
    (filtered by `event_regex`, `min_nearby`, `max_within`) and a `KernelManager`
    to retrieve the actual multi-modal data (like images, actions, etc.) for a window
    of time (`win_len`) around each event. It supports splitting into training and
    validation sets.

    :param dataset_dirs: List of directories where the raw dataset (LMDBs for different modalities)
                         is stored.
    :type dataset_dirs: List[str]
    :param modal_kernel_callbacks: List of `ModalKernelCallback` instances or their registered names.
                                   These define how data for each modality is fetched and processed.
    :type modal_kernel_callbacks: List[Union[str, ModalKernelCallback]]
    :param modal_kernel_config: Optional configuration dictionary for creating `ModalKernelCallback`
                                instances if their names are provided. Defaults to None.
    :type modal_kernel_config: Optional[Dict]
    :param win_len: The length of the window (number of frames/steps) to retrieve around each event.
                    Defaults to 1.
    :type win_len: int
    :param skip_frame: The number of frames to skip between consecutive frames in the window.
                       Defaults to 1 (no skip).
    :type skip_frame: int
    :param split: Specifies whether this dataset instance is for 'train' or 'val'.
                  Defaults to 'train'.
    :type split: Literal['train', 'val']
    :param split_ratio: The ratio of data to be used for the training set. The rest is for validation.
                        Defaults to 0.8.
    :type split_ratio: float
    :param verbose: If True, logs information during setup. Defaults to True.
    :type verbose: bool
    :param event_paths: Optional list of paths to event LMDB databases. If None, it assumes
                        event databases are in an "event" subdirectory within each `dataset_dirs`.
                        Defaults to None.
    :type event_paths: Optional[List[str]]
    :param bias: An offset applied to the start time when fetching the window of data around an event.
                 `start = max(event_time - win_len + bias, 0)`. Defaults to 0.
    :type bias: int
    :param event_regex: Regular expression to filter event names from the `EventKernelManager`.
                        Defaults to ''.
    :type event_regex: str
    :param min_nearby: Passed to `EventKernelManager` to filter events. Minimum time between
                       consecutive events of the same type. Defaults to None.
    :type min_nearby: Optional[int]
    :param max_within: Passed to `EventKernelManager` to filter events. Maximum number of instances
                       to keep for each event type. Defaults to None.
    :type max_within: Optional[int]
    """
    
    def __init__(self, 
        dataset_dirs: List[str], 
        modal_kernel_callbacks: List[Union[str, ModalKernelCallback]], 
        modal_kernel_config: Optional[Dict]=None,
        # below are parameters for spliting dataset and building items
        win_len: int=1, 
        skip_frame: int=1,
        split: Literal['train', 'val']='train',
        split_ratio: float=0.8, 
        verbose: bool=True,
        # below are event dataset specific parameters
        event_paths: Optional[List[str]]=None,
        bias: int=0,
        event_regex: str='', 
        min_nearby: Optional[int]=None, # minimal avaliable distance between two selected events
        max_within: Optional[int]=None, # maximum number of samples within each event
    ) -> Any:
        super().__init__()
        self.win_len = win_len
        self.skip_frame = skip_frame
        self.split = split
        self.split_ratio = split_ratio
        self.verbose = verbose
        self.bias = bias
        self.event_regex = event_regex

        if event_paths is None:
            event_paths = [Path(x) / "event" for x in dataset_dirs]
        else:
            event_paths = [Path(x) for x in event_paths]

        self.event_kernel = EventKernelManager(
            event_path=event_paths,
            event_regex=event_regex,
            verbose=verbose,
            min_nearby=min_nearby, 
            max_within=max_within,
        )
        
        assert len(modal_kernel_callbacks) > 0, "At least one modal kernel callback is required. "
        if isinstance(modal_kernel_callbacks[0], str):
            assert modal_kernel_config is not None, "Modal kernel config is required. "
            modal_kernel_callbacks = [
                Registers.modal_kernel_callback[name].create_from_config(modal_kernel_config) 
                    for name in modal_kernel_callbacks
            ]
        
        self.kernel_manager = KernelManager(
            dataset_dirs=dataset_dirs, 
            modal_kernel_callbacks=modal_kernel_callbacks,
        )
        
        self.build_items()
    
    def build_items(self) -> None:
        """
        Builds the list of items for the dataset based on selected events and split.

        This method populates `self.items`, which is a list of tuples.
        Each tuple stores a cumulative count of items, the event name, and a bias
        for indexing into the original event list (used for train/val splitting).
        The total number of items in the dataset (`self.num_items`) is also calculated.
        """
        self.event_list = self.event_kernel.get_event_list()
        
        self.num_items = 0
        event_with_items = []
        for event in self.event_list:
            num_event_items = self.event_kernel.get_event_size(event)
            if self.split == 'train':
                bias = 0
                num_event_items = int(num_event_items * self.split_ratio)
            elif self.split == 'val':
                bias = int(num_event_items * self.split_ratio)
                num_event_items = num_event_items - bias
            else:
                raise ValueError(f"Split type <{self.split}> not supported. ")
            self.num_items += num_event_items
            event_with_items.append((self.num_items, event, bias))
        self.items = event_with_items
        
        if self.verbose:
            Console().log(f"[Event Dataset] Regex: {self.event_regex}, Number of events: {len(self.event_list)}, number of items: {self.num_items}")
    
    def locate_item(self, idx: int) -> Tuple[str, int]:
        """
        Locates the specific event and relative index within that event for a given global item index.

        It uses a binary search on `self.items` (which stores cumulative counts)
        to efficiently find which event the `idx` falls into and what its relative
        index within that event's items is (after considering the split bias).

        :param idx: The global index of the item in the dataset.
        :type idx: int
        :returns: A tuple containing:
                  - event (str): The name of the event.
                  - relative_idx_with_bias (int): The index within the specific event's item list,
                                                  adjusted for the train/val split bias.
        :rtype: Tuple[str, int]
        """
        # Find the first event that idx > acc[event]
        left, right = 0, len(self.items)
        while left < right:
            mid = (left + right) // 2
            if self.items[mid][0] <= idx:
                left = mid + 1
            else:
                right = mid
        if left == 0:
            relative_idx = idx
        else:
            relative_idx = idx - self.items[left-1][0]
        event = self.items[left][1]
        bias = self.items[left][2]
        return event, relative_idx + bias
    
    def __len__(self) -> int:
        """
        Returns the total number of items in this dataset instance (train or val split).

        :returns: The number of items.
        :rtype: int
        """
        return self.num_items
    
    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        """
        Retrieves a single data item (a window of multi-modal data centered around an event).

        1.  Locates the event and its relative index using `locate_item`.
        2.  Gets the specific event details (episode, time) using `self.event_kernel.get_event_item`.
        3.  Calculates the start time for the data window.
        4.  Reads the multi-modal data for the window using `self.kernel_manager.read`.
        5.  Extracts any mask associated with the item.
        6.  Adds event text, episode name, and timestamps to the item.
        7.  Converts all NumPy arrays in the item to PyTorch tensors using `to_tensor`.

        :param idx: The global index of the item to retrieve.
        :type idx: int
        :raises AssertionError: If `idx` is out of range.
        :returns: A mapping where keys are modality names (or 'text', 'episode', 'timestamp', 'mask')
                  and values are the corresponding data as PyTorch tensors.
        :rtype: Mapping[str, torch.Tensor]
        """
        assert idx < len(self), f"Index <{idx}> out of range <{len(self)}>"
        event, relative_idx = self.locate_item(idx)
        episode, event_time, value = self.event_kernel.get_event_item(event, relative_idx)
        start = max(event_time - self.win_len + self.bias, 0)
        item = self.kernel_manager.read(episode, start=start, win_len=self.win_len, skip_frame=self.skip_frame)

        for key in list(item.keys()):
            if key.endswith('mask'):
                mask = item.pop(key)
        item["mask"] = mask

        item['text'] = event.replace('minecraft.', '')
        item['episode'] = episode
        item['timestamp'] = np.arange(start, start+self.win_len, self.skip_frame)
        item = self.to_tensor(item)
        return item

    def to_tensor(self, item: Union[np.ndarray, List, Dict]) -> Union[torch.Tensor, List, Dict]:
        """Recursively converts NumPy arrays within a nested structure (list or dict) to PyTorch tensors.

        If the input item is a NumPy array, it's converted to a PyTorch tensor.
        If it's a list, the conversion is applied to each element.
        If it's a dictionary, the conversion is applied to each value.
        Other types are returned as is.

        :param item: The item to convert. Can be a NumPy array, or a list/dict containing them.
        :type item: Union[np.ndarray, List, Dict]
        :returns: The item with NumPy arrays replaced by PyTorch tensors.
        :rtype: Union[torch.Tensor, List, Dict]
        """
        # Convert numpy array to torch tensor.
        if isinstance(item, np.ndarray):
            tensor = torch.from_numpy(item)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            return tensor
        elif isinstance(item, List):
            return [self.to_tensor(val) for val in item]
        elif isinstance(item, Dict):
            return {key: self.to_tensor(val) for key, val in item.items()}
        else:
            return item


class EventDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling event-based datasets.

    This class encapsulates the train and validation `EventDataset` instances and provides
    corresponding DataLoaders. It simplifies the data handling pipeline for training
    and evaluating models with PyTorch Lightning.

    :param data_params: Dictionary of parameters to be passed to the `EventDataset` constructor.
                        This includes `dataset_dirs`, `modal_kernel_callbacks`, `win_len`, etc.
    :type data_params: Dict
    :param batch_size: The batch size for the DataLoaders, defaults to 1.
    :type batch_size: int, optional
    :param num_workers: The number of worker processes for data loading, defaults to 0.
    :type num_workers: int, optional
    :param prefetch_factor: Number of batches loaded in advance by each worker. Defaults to None.
                            See PyTorch DataLoader documentation for more details.
    :type prefetch_factor: Optional[int]
    """
    
    def __init__(self, 
                 data_params: Dict, 
                 batch_size: int=1, 
                 num_workers: int=0, 
                 prefetch_factor: Optional[int] = None):
        super().__init__()
        self.data_params = data_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
    
    def setup(self, stage: Optional[str]=None):
        """
        Sets up the training and validation datasets.

        This method is called by PyTorch Lightning at the appropriate time (e.g., before training
        or validation). It instantiates `EventDataset` for both 'train' and 'val' splits
        using the provided `data_params`.

        :param stage: A string indicating the current stage (e.g., 'fit', 'validate', 'test').
                      Not directly used in this implementation but part of the Lightning interface.
                      Defaults to None.
        :type stage: Optional[str]
        """
        self.train_dataset = EventDataset(split='train', **self.data_params)
        self.val_dataset = EventDataset(split='val', **self.data_params)

    def train_dataloader(self):
        """
        Sets up the DataLoader for the training dataset.
        """
        train_loader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True, 
            collate_fn=batchify,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        """
        Sets up the DataLoader for the validation dataset.
        """
        val_loader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            collate_fn=batchify,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            drop_last=True,
        )
        return val_loader


if __name__ == '__main__':

    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from minestudio.data.minecraft.callbacks import (
        ImageKernelCallback, ActionKernelCallback, MetaInfoKernelCallback, SegmentationKernelCallback
    )


    data_module = EventDataModule(
        data_params=dict(
            dataset_dirs=[
                '/nfs-shared-2/data/contractors/dataset_10xx', 
            ], 
            modal_kernel_callbacks=[
                ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False), 
                ActionKernelCallback(),
                MetaInfoKernelCallback(),
                SegmentationKernelCallback(frame_width=224, frame_height=224), 
            ],
            win_len=128, 
            split_ratio=0.9, 
            event_regex='minecraft.kill_entity:.*', 
            min_nearby=64,
            max_within=1000,
        ), 
        batch_size=4, 
        num_workers=4, 
        prefetch_factor=None
    )
    data_module.setup()
    loader = data_module.train_dataloader()
    for idx, batch in enumerate(loader):
        print(
            "\t".join(
                [f"{a} {b}" for a, b in zip(batch['episode'], batch['text'])]
            )
        )
        if idx > 50:
            break
