'''
Date: 2025-01-09 05:45:49
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-10 14:52:41
FilePath: /MineStudio/minestudio/data/minecraft/core.py
'''
import lmdb
import pickle
import numpy as np
from rich import print
from rich.console import Console
from collections import OrderedDict
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Sequence, Mapping, Any, Optional, Literal

from minestudio.data.minecraft.callbacks import ModalKernelCallback
from minestudio.data.minecraft.utils import pull_datasets_from_remote

class ModalKernel(object):

    SHORT_NAME_LENGTH = 8

    def __init__(self, source_dirs: List[str], modal_kernel_callback: ModalKernelCallback, short_name: bool = False):
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
        return self.modal_kernel_callback.name
    
    def read_chunks(self, eps: str, start: int, end: int) -> List[bytes]:
        """
        Given episode name and required interval, return the corresponding chunks.
        [start, end] refer to frame-level indexes, which % self.chunk_size == 0. 
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
    
    def read_frames(self, eps: str, start: int, win_len: int, skip_frame: int) -> Dict:
        """
        Given episode name and required interval, return the corresponding frames.
        [start, end] refer to a frame-level index, 0 <= start <= end < num_frames
        """
        meta_info = self.episode_infos[self.eps_idx_mapping[eps]]
        end = min(start + win_len * skip_frame - 1, meta_info['num_frames'] - 1) # include
        chunk_bytes = self.read_chunks(eps, 
            start // self.chunk_size * self.chunk_size, 
            end // self.chunk_size * self.chunk_size
        )
        # 1. merge chunks into continuous frames
        frames = self.modal_kernel_callback.do_merge(chunk_bytes)
        # 2. extract frames according to skip_frame
        bias = (start // self.chunk_size) * self.chunk_size
        frames = self.modal_kernel_callback.do_slice(frames, start - bias, end - bias + 1, skip_frame)
        # 3. padding frames and get masks
        frames, mask = self.modal_kernel_callback.do_pad(frames, win_len)
        result = { f"{self.name}": frames, f"{self.name}_mask": mask }
        # 4. do postprocess
        result = self.modal_kernel_callback.do_postprocess(result)
        return result
    
    def get_episode_list(self) -> List[str]:
        return [info['episode'] for info in self.episode_infos]
    
    def get_num_frames(self, episodes: Optional[List[str]] = None):
        if episodes is None:
            episodes = self.eps_idx_mapping.keys()
        num_frames = 0
        for eps in episodes:
            info_idx = self.eps_idx_mapping[eps]
            num_frames += self.episode_infos[info_idx]['num_frames']
        return num_frames

class KernelManager(object):

    def __init__(self, dataset_dirs: List[str], modal_kernel_callbacks: List[ModalKernelCallback], verbose: bool = True):
        """The kernel class for managing datasets. It provides a unified interface for 
        accessing demonstrations information such as video, action, and meta_info. 
        Args:
            dataset_dirs (List[str]): A list of paths to dataset directories. 
                It is supposed the dataset directory contains the following subdirectories:
                video, action, meta_info, ...
            modal_kernel_callbacks (List[ModalKernelCallback]): A list of modal kernel callbacks.
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

    def read(self, eps: str, start: int, win_len: int, skip_frame: int) -> Dict:
        """Read all avaliable modals from lmdb files."""
        result = {}
        for modal, kernel in self.kernels.items():
            modal_result = kernel.read_frames(eps, start, win_len, skip_frame)
            result.update(modal_result)
            # if modal == 'action':
            #     prev_frames, prev_mask = self.read_frames(eps, start-1, win_len, skip_frame, modal) # start must > 0
            #     result[f'prev_action'] = prev_frames
            #     result[f'prev_action_mask'] = prev_mask
        return result

    def get_num_frames(self):
        return self.num_frames

    def get_episodes_with_length(self):
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