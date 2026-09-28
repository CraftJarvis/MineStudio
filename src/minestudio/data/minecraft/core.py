'''
Date: 2025-01-09 05:45:49
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-03-17 21:54:30
FilePath: /MineStudio/minestudio/data/minecraft/core.py
'''
import lmdb
import pickle
import hashlib
import numpy as np
from rich import print
from rich.console import Console
from collections import OrderedDict
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Sequence, Mapping, Any, Optional, Literal

from minestudio.data.minecraft.callbacks import ModalKernelCallback
from minestudio.data.minecraft.utils import pull_datasets_from_remote

class ModalKernel(object):
    """
    Manages and provides access to data for a single modality (e.g., video, actions) 
    from a collection of LMDB datasets. It merges metadata and provides methods 
    to read chunks and frames of data for specific episodes.

    :param source_dirs: A list of directory paths, each containing LMDB files for the modality.
    :type source_dirs: List[str]
    :param modal_kernel_callback: A callback object to handle modality-specific operations 
                                  like data merging, slicing, and padding.
    :type modal_kernel_callback: ModalKernelCallback
    :param short_name: If True, episode names are hashed to a shorter length. Defaults to False.
    :type short_name: bool, optional
    """

    SHORT_NAME_LENGTH = 8

    def __init__(self, source_dirs: List[str], modal_kernel_callback: ModalKernelCallback, short_name: bool = False):
        """
        Initializes the ModalKernel by merging metadata from multiple LMDB datasets.

        It iterates through each source directory, opens LMDB files, and aggregates 
        episode information, total frames, and chunk size. It also creates a mapping 
        from episode names to their indices for quick lookups.

        :param source_dirs: A list of directory paths, each containing LMDB files for the modality.
        :type source_dirs: List[str]
        :param modal_kernel_callback: A callback object to handle modality-specific operations.
        :type modal_kernel_callback: ModalKernelCallback
        :param short_name: If True, episode names are hashed. Defaults to False.
        :type short_name: bool, optional
        """
        super().__init__()
        self.modal_kernel_callback = modal_kernel_callback
        source_dirs = self.modal_kernel_callback.filter_dataset_paths(source_dirs)
        self.episode_infos = []
        self.num_episodes = 0
        self.num_total_frames = 0
        self.chunk_size = None
        # merge all lmdb files into one single view
        for source_dir in source_dirs:
            for lmdb_path in source_dir.iterdir():
                stream = lmdb.open(str(lmdb_path), max_readers=128, lock=False, readonly=True)
                # self.lmdb_streams.append(stream)
                with stream.begin() as txn:
                    # read meta infos from each lmdb file
                    __chunk_size__ = pickle.loads(txn.get("__chunk_size__".encode()))
                    __chunk_infos__ = pickle.loads(txn.get("__chunk_infos__".encode()))
                    __num_episodes__ = pickle.loads(txn.get("__num_episodes__".encode()))
                    __num_total_frames__ = pickle.loads(txn.get("__num_total_frames__".encode()))
                    # merge meta infos to a single view
                    for chunk_info in __chunk_infos__:
                        chunk_info['lmdb_stream'] = stream
                        if short_name:
                            chunk_info['episode'] = hashlib.md5(chunk_info['episode'].encode()).hexdigest()[:SHORT_NAME_LENGTH]
                    self.episode_infos += __chunk_infos__
                    self.num_episodes += __num_episodes__
                    self.num_total_frames += __num_total_frames__
                    self.chunk_size = __chunk_size__
        # create a episode to index mapping 
        self.eps_idx_mapping = { info['episode']: idx for idx, info in enumerate(self.episode_infos) }
    
    @property
    def name(self):
        """
        Returns the name of the modality, as defined by the modal_kernel_callback.

        :returns: The name of the modality.
        :rtype: str
        """
        return self.modal_kernel_callback.name
    
    def read_chunks(self, eps: str, start: int, end: int) -> List[bytes]:
        """
        Reads and returns a list of data chunks for a given episode and frame range.

        The start and end parameters specify the frame-level indices, which must be 
        multiples of the chunk_size.

        :param eps: The name of the episode to read from.
        :type eps: str
        :param start: The starting frame index (inclusive, multiple of chunk_size).
        :type start: int
        :param end: The ending frame index (inclusive, multiple of chunk_size).
        :type end: int
        :returns: A list of byte strings, where each string is a data chunk.
        :rtype: List[bytes]
        :raises AssertionError: If start or end are not multiples of chunk_size.
        """
        assert start % self.chunk_size == 0 and end % self.chunk_size == 0
        meta_info = self.episode_infos[self.eps_idx_mapping[eps]]
        read_chunks = []
        for chunk_id in range(start, end + self.chunk_size, self.chunk_size):
            with meta_info['lmdb_stream'].begin() as txn:
                key = str((meta_info['episode_idx'], chunk_id)).encode()
                chunk_bytes = txn.get(key)
                read_chunks.append(chunk_bytes)

        return read_chunks
    
    def read_frames(self, eps: str, start: int, win_len: int, skip_frame: int, **kwargs) -> Dict:
        """
        Reads, processes, and returns a dictionary of frames for a given episode and window.

        This method handles reading data chunks, merging them into continuous frames, 
        slicing based on skip_frame, and padding if necessary. It utilizes the 
        modal_kernel_callback for modality-specific operations.

        :param eps: The name of the episode.
        :type eps: str
        :param start: The starting frame index.
        :type start: int
        :param win_len: The desired window length (number of frames).
        :type win_len: int
        :param skip_frame: The number of frames to skip between selected frames.
        :type skip_frame: int
        :param \\**kwargs: Additional arguments passed to the modal_kernel_callback.
        :returns: A dictionary containing the processed frames and a corresponding mask. 
                  The keys are formatted as "{modality_name}" and "{modality_name}_mask".
        :rtype: Dict
        """
        meta_info = self.episode_infos[self.eps_idx_mapping[eps]]
        start += self.modal_kernel_callback.read_bias #! adding read_bias to the original range
        win_len += self.modal_kernel_callback.win_bias #! adding win_bias to the original range
        end = min(start + win_len * skip_frame - 1, meta_info['num_frames'] - 1) # include
        
        if start >= 0:
            pad_left = 0
        else:
            pad_left = -start
            start = 0
        
        chunk_bytes = self.read_chunks(eps, 
            start // self.chunk_size * self.chunk_size, 
            end // self.chunk_size * self.chunk_size, 
        )
        # 1. merge chunks into continuous frames
        frames = self.modal_kernel_callback.do_merge(chunk_bytes, **kwargs)
        # 2. extract frames according to skip_frame
        bias = (start // self.chunk_size) * self.chunk_size
        frames = self.modal_kernel_callback.do_slice(frames, start - bias, end - bias + 1, skip_frame, **kwargs)
        mask = np.ones(end-start+1, dtype=np.uint8)
        # 3. padding frames and get masks
        # -> 3.1 padding left
        if pad_left > 0:
            frames, mask = self.modal_kernel_callback.do_pad(frames, pad_left, "left", **kwargs)
        # -> 3.2 padding right
        if win_len - len(mask) > 0:
            frames, right_mask = self.modal_kernel_callback.do_pad(frames, win_len - len(mask), "right", **kwargs)
            mask = np.concatenate([mask, right_mask[len(mask):]], axis=0)
        result = { f"{self.name}": frames, f"{self.name}_mask": mask }
        # 4. do postprocess
        result = self.modal_kernel_callback.do_postprocess(result)
        return result
    
    def get_episode_list(self) -> List[str]:
        """
        Returns a list of all episode names managed by this kernel.

        :returns: A list of episode names.
        :rtype: List[str]
        """
        return [info['episode'] for info in self.episode_infos]
    
    def get_num_frames(self, episodes: Optional[List[str]] = None):
        """
        Calculates and returns the total number of frames for the specified episodes.

        If no episodes are provided, it calculates the total frames for all episodes 
        managed by this kernel.

        :param episodes: An optional list of episode names. If None, all episodes are considered.
        :type episodes: Optional[List[str]], optional
        :returns: The total number of frames.
        :rtype: int
        """
        if episodes is None:
            episodes = self.eps_idx_mapping.keys()
        num_frames = 0
        for eps in episodes:
            info_idx = self.eps_idx_mapping[eps]
            num_frames += self.episode_infos[info_idx]['num_frames']
        return num_frames

class KernelManager(object):
    """
    Manages multiple ModalKernel instances, providing a unified interface for accessing 
    data from different modalities (e.g., video, actions, metadata) in a dataset.

    It loads and organizes data from specified dataset directories, ensuring consistency 
    across modalities and episodes.

    :param dataset_dirs: A list of paths to dataset directories. Each directory is expected 
                         to contain subdirectories for different modalities.
    :type dataset_dirs: List[str]
    :param modal_kernel_callbacks: A list of ModalKernelCallback objects, one for each 
                                   modality to be managed.
    :type modal_kernel_callbacks: List[ModalKernelCallback]
    :param verbose: If True, prints logging information during initialization. Defaults to True.
    :type verbose: bool, optional
    """

    def __init__(self, dataset_dirs: List[str], modal_kernel_callbacks: List[ModalKernelCallback], verbose: bool = True):
        """
        Initializes the KernelManager by setting up dataset directories and loading modal kernels.

        It first pulls datasets from remote sources if necessary, then identifies 
        sub-dataset directories. Finally, it calls `load_modal_kernels` to initialize 
        kernels for each specified modality.

        :param dataset_dirs: A list of paths to dataset directories.
        :type dataset_dirs: List[str]
        :param modal_kernel_callbacks: A list of ModalKernelCallback objects.
        :type modal_kernel_callbacks: List[ModalKernelCallback]
        :param verbose: If True, enables verbose logging. Defaults to True.
        :type verbose: bool, optional
        """
        super().__init__()
        dataset_dirs = pull_datasets_from_remote(dataset_dirs)
        sub_dataset_dirs = []
        for str_dir in sorted(dataset_dirs):
            for sub_dir in Path(str_dir).iterdir():
                sub_dataset_dirs.append(sub_dir)
        self.sub_dataset_dirs = sub_dataset_dirs
        self.modal_kernel_callbacks = modal_kernel_callbacks
        self.verbose = verbose
        self.load_modal_kernels()

    def load_modal_kernels(self):
        """
        Loads a ModalKernel for each modality specified in modal_kernel_callbacks.

        It iterates through the callbacks, creates a ModalKernel for each, and stores 
        them in the `kernels` dictionary. It also determines the common episodes 
        across all modalities and calculates the total number of frames.
        """
        self.kernels = dict()
        episodes = None

        for modal_kernel_callback in self.modal_kernel_callbacks:
            kernel = ModalKernel(self.sub_dataset_dirs, modal_kernel_callback, short_name=False)
            self.kernels[kernel.name] = kernel
            part_episodes = set(kernel.get_episode_list())
            if self.verbose:
                Console().log(f"[Kernel] Modal [pink]{kernel.name}[/pink] load {len(part_episodes)} episodes. ")
            episodes = episodes.intersection(part_episodes) if episodes is not None else part_episodes

        self.num_frames = 0
        self.episodes_with_length = OrderedDict()
        for episode in sorted(list(episodes)):
            num_list = [kernel.get_num_frames([episode]) for kernel in self.kernels.values()]
            if len(set(num_list)) != 1:
                continue
            self.num_frames += num_list[0]
            self.episodes_with_length[episode] = num_list[0]

        if self.verbose:
            Console().log(f"[Kernel] episodes: {len(self.episodes_with_length)}, frames: {self.num_frames}. ")

    def read(self, eps: str, start: int, win_len: int, skip_frame: int, **kwargs) -> Dict:
        """
        Reads and returns data for all managed modalities for a given episode and window.

        It iterates through each loaded kernel, calls its `read_frames` method, 
        and aggregates the results into a single dictionary.

        :param eps: The name of the episode.
        :type eps: str
        :param start: The starting frame index.
        :type start: int
        :param win_len: The desired window length (number of frames).
        :type win_len: int
        :param skip_frame: The number of frames to skip between selected frames.
        :type skip_frame: int
        :param \\**kwargs: Additional arguments passed to the `read_frames` method of each kernel.
        :returns: A dictionary containing data from all modalities for the specified window.
        :rtype: Dict
        """
        result = {}
        for modal, kernel in self.kernels.items():
            # if modal != 'meta_info': continue
            modal_result = kernel.read_frames(eps, start, win_len, skip_frame, **kwargs)
            result.update(modal_result)
        return result

    def get_num_frames(self):
        """
        Returns the total number of frames across all common episodes and modalities.

        :returns: The total number of frames.
        :rtype: int
        """
        return self.num_frames

    def get_episodes_with_length(self):
        """
        Returns an OrderedDict mapping common episode names to their lengths (number of frames).

        :returns: An OrderedDict where keys are episode names and values are their lengths.
        :rtype: OrderedDict
        """
        return self.episodes_with_length


if __name__ == "__main__":
    from minestudio.data.minecraft.callbacks import ImageKernelCallback, ActionKernelCallback, MetaInfoKernelCallback
    kernel_manager = KernelManager(
        dataset_dirs=[
            '/nfs-shared-2/data/contractors/dataset_10xx', 
        ], 
        modal_kernel_callbacks=[
            ImageKernelCallback(frame_width=128, frame_height=128, enable_video_aug=True),
            ActionKernelCallback(), 
            MetaInfoKernelCallback(),
        ],
    )
    episodes_with_length = kernel_manager.get_episodes_with_length()
    for eps, length in episodes_with_length.items():
        result = kernel_manager.read(eps, 0, 128, 1)
        print(result.keys())
        break