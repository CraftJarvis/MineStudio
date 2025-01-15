'''
Date: 2024-11-10 12:27:01
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-15 15:04:05
FilePath: /MineStudio/minestudio/data/minecraft/tools/convertion.py
'''

import ray
import time
import pickle
import ray.experimental.tqdm_ray as tqdm_ray

import lmdb
import numpy as np
import shutil
from tqdm import tqdm
from rich import print
from rich.console import Console
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any
from collections import OrderedDict

from minestudio.data.minecraft.callbacks import ModalConvertCallback

@ray.remote(num_cpus=1)
class ConvertWorker:
    
    def __init__(
        self, 
        write_path: Union[str, Path], 
        convert_kernel: ModalConvertCallback,
        tasks: Dict, 
        chunk_size: int,
        remote_tqdm: Any, 
        thread_pool: int = 8,
        filter_kernel: Optional[ModalConvertCallback]=None,
    ) -> None:
        self.tasks = tasks
        self.write_path = write_path
        self.chunk_size = chunk_size
        self.remote_tqdm = remote_tqdm
        self.thread_pool = thread_pool
        self.convert_kernel = convert_kernel
        self.filter_kernel = filter_kernel

        if isinstance(write_path, str):
            write_path = Path(write_path) 
        if write_path.is_dir():
            print(f"Write path {write_path} exists, delete it. ")
            shutil.rmtree(write_path)
        write_path.mkdir(parents=True)
        self.lmdb_handler = lmdb.open(str(write_path), map_size=1<<40)


    def run(self):
        chunk_infos = []
        num_total_frames = 0
        eps_idx = 0
        for eps, parts in self.tasks.items():
            # if eps_idx > 3: #! debug !!!
            #     break
            eps_keys, eps_vals = [], []
            eps_keys, eps_vals, cost = self.convert(eps=eps, parts=parts)
            num_eps_frames = len(eps_keys) * self.chunk_size
            if num_eps_frames == 0:
                # empty video, skip it
                continue
            for key, val in zip(eps_keys, eps_vals):
                with self.lmdb_handler.begin(write=True) as txn:
                    lmdb_key = str((eps_idx, key))
                    txn.put(str(lmdb_key).encode(), val)
            chunk_info = {
                "episode": eps, 
                "episode_idx": eps_idx,
                "num_frames": num_eps_frames,
            }
            chunk_infos.append(chunk_info)
            num_total_frames += num_eps_frames
            eps_idx += 1

            self.remote_tqdm.update.remote(1)

        meta_info = {
            "__chunk_size__": self.chunk_size,
            "__chunk_infos__": chunk_infos,
            "__num_episodes__": eps_idx,
            "__num_total_frames__": num_total_frames,
        }
        
        with self.lmdb_handler.begin(write=True) as txn:
            for key, val in meta_info.items():
                txn.put(key.encode(), pickle.dumps(val))
        
        print(f"Worker finish: {self.write_path}. ")
        return meta_info

    def convert(self, eps: str, parts: List[Tuple[int, Path, Path]]) -> Tuple[List, List, float]:
        time_start = time.time()
        skip_frames = []
        modal_file_path = []
        for i in range(len(parts)):
            modal_file_path.append(parts[i][1])
            if self.filter_kernel is not None:
                file_name = parts[i][1].stem
                skip_frames.append(self.filter_kernel.gen_frame_skip_flags(file_name))
            else:
                skip_frames.append( None )
        keys, vals = self.convert_kernel.do_convert(eps, skip_frames, modal_file_path)
        cost = time.time() - time_start
        print(f"episode: {eps}, chunks: {len(keys)}, frames: {len(keys) * self.chunk_size}, "
                f"size: {sum(len(x) for x in vals) / (1024*1024):.2f} MB, cost: {cost:.2f} sec")
        return keys, vals, cost


class ConvertManager:

    def __init__(
        self, 
        output_dir: str,
        convert_kernel: ModalConvertCallback,
        filter_kernel: Optional[ModalConvertCallback]=None,
        chunk_size: int=32, 
        num_workers: int=16,
    ) -> None:
        self.output_dir = output_dir
        self.convert_kernel = convert_kernel
        self.filter_kernel = filter_kernel
        self.chunk_size = chunk_size
        self.num_workers = num_workers

    def prepare_tasks(self):
        source_episodes = self.convert_kernel.load_episodes()
        if self.filter_kernel is not None:
            filter_episodes = self.filter_kernel.load_episodes()
        loaded_episodes = OrderedDict()
        num_removed_parts = 0
        for eps, source_parts in source_episodes.items():
            # 1. check if the episode is in the filter list
            if self.filter_kernel is not None and eps not in filter_episodes:
                num_removed_parts += len(source_parts)
                continue
            
            for ord, source_path in source_parts:
                # 2. check if the part is in the filter list
                if self.filter_kernel is not None:
                    intersection = [part for part in filter_episodes[eps] if part[0] == ord]
                    if len(intersection) == 0:
                        num_removed_parts += 1
                        continue
                if eps not in loaded_episodes:
                    loaded_episodes[eps] = []
                loaded_episodes[eps].append( (ord, source_path) )
        self.loaded_episodes = loaded_episodes
        print(f"[ConvertManager] num of removed episode parts: {num_removed_parts}")

    def dispatch(self):
        sub_tasks = OrderedDict()
        workers = []
        remote_tqdm = ray.remote(tqdm_ray.tqdm).remote(total=len(self.loaded_episodes))
        num_episodes_per_file = (len(self.loaded_episodes) + self.num_workers - 1) // self.num_workers
        for idx, (eps, parts) in enumerate(self.loaded_episodes.items()):
            sub_tasks[eps] = parts
            if (idx + 1) % num_episodes_per_file == 0 or (idx + 1) == len(self.loaded_episodes):
                write_path = Path(self.output_dir) / f"part-{idx+1}"
                worker = ConvertWorker.remote(
                    write_path=write_path, 
                    convert_kernel=self.convert_kernel, 
                    tasks=sub_tasks, 
                    chunk_size=self.chunk_size, 
                    remote_tqdm=remote_tqdm, 
                    filter_kernel=self.filter_kernel,
                )
                workers.append(worker)
                sub_tasks = OrderedDict()
        results = ray.get([worker.run.remote() for worker in workers])
        num_frames   = sum([result['__num_total_frames__'] for result in results])
        num_episodes = sum([result['__num_episodes__'] for result in results])
        ray.kill(remote_tqdm)
        print(f"Total frames: {num_frames}, Total episodes: {num_episodes}")
