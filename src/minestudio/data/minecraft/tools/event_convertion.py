'''
Date: 2024-11-10 12:26:39
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-03-17 21:40:14
FilePath: /MineStudio/var/minestudio/data/minecraft/tools/event_convertion.py
'''

import re
import os
import lmdb
import time
import random
import pickle
import argparse
import shutil
from collections import OrderedDict

import math 
import torch
import numpy as np

import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from rich import print
from rich.console import Console
from pathlib import Path
from typing import Union, Tuple, List, Dict

from minestudio.data.minecraft.core import KernelManager
from minestudio.data.minecraft.callbacks import MetaInfoKernelCallback, ImageKernelCallback, ActionKernelCallback

'''
    Desired data structure of lmdb files: 
    {
        '__codebook__': {'eps1': '0', 'eps2': '1'}, (could be omitted) 
        '__num_events__': 600, 
        '__event_info__': {
            'pickup:mutton': {
                '__num_items__': 500,
                '__num_episodes__': 20,
            }, 
            'mine_block:grass': {
                '__num_items__': 450,
                '__num_episodes__': 16,
            }, 
            ...
        }, 
        '(pickup:mutton, 0)': (episode_xxx, t1, v1), 
        '(pickup:mutton, 1)': (episode_xxx, t2, v2), 
        '(pickup:mutton, 2)': (episode_yyy, t1, v1),
        '(mine_block:grass, 0)': (episode_zzz, t1, v1),
        ...
    }
'''

def main(args):
    """
    Main function to process Minecraft game events and store them in an LMDB database.

    This function reads event data from a specified input directory, processes it using
    a KernelManager, and then organizes and writes the events into an LMDB database
    with a specific structure. The structure includes a codebook for episode IDs,
    the total number of events, information about each event type (number of items
    and episodes), and individual event occurrences.

    :param args: Command-line arguments, expected to have an `input_dir` attribute
                 specifying the directory containing the raw game data.
    :type args: argparse.Namespace
    """
    
    event_path = Path(args.input_dir) / "event"
    if event_path.is_dir():
        print(f"Directory {event_path} exists, remove and recreate one. ")
        shutil.rmtree(event_path)
    event_path.mkdir(parents=True)
    
    meta_info_path = Path(args.input_dir) / 'meta_info'
    assert meta_info_path.is_dir(), f"Directory {meta_info_path} does not exist. "
    
    kernel = KernelManager(
        dataset_dirs=[args.input_dir], 
        modal_kernel_callbacks=[
            ImageKernelCallback(
                frame_width=224, 
                frame_height=224, 
            ), 
            ActionKernelCallback(),
            MetaInfoKernelCallback(),
        ],
    )
    
    episode_with_length = kernel.get_episodes_with_length()
    episodes = [x for x in episode_with_length.keys()]
    
    events = {}
    # monitor_fields = ['delta_craft_item', 'delta_mine_block', 'delta_pickup']
    monitor_fields = ['events']
    import ipdb;
    for idx, episode in enumerate(tqdm(episodes)):
        length = episode_with_length[episode]
        result = kernel.read(episode, start=0, win_len=length, skip_frame=1)
        frames, mask = result['meta_info'], result['meta_info_mask']
        assert mask.sum() == length, f"Mask sum: {mask.sum()}, length: {length}. "
        # enumerate all fields of interest and generate all the events
        for field in monitor_fields:
            records: List[Dict] = frames[field]
            for t, record in enumerate(records):
                if len(record) == 0:
                    continue
                for event, value in record.items():
                    if event not in events:
                        events[event] = {}
                    if episode not in events[event]:
                        events[event][episode] = []
                    events[event][episode].append( (t, value) )
    
    # write events into lmdb files in the desired structure
    lmdb_data = {
        '__num__events__': len(events), 
        '__event_info__': {}, 
    }
    
    print("Total events:", lmdb_data['__num__events__'])
    
    for event, episode_items in events.items():
        lmdb_data['__event_info__'][event] = {
            '__num_episodes__': len(episode_items),
            '__num_items__': sum([len(x) for x in episode_items.values()]),
        }
    
    codebook = {}
    for event, episode_items in events.items():
        event_item_id = 0
        for episode, items in episode_items.items():
            # update codebook
            if episode not in codebook:
                    codebook[episode] = f"{len(codebook)}"
            for e_time, value in items:
                key = str((event, event_item_id))
                # triple = (episode, e_time, value)
                triple = (codebook[episode], e_time, value)
                lmdb_data[key] = triple
                event_item_id += 1
    lmdb_data['__codebook__'] = codebook
    
    with lmdb.open(str(event_path), map_size=1<<40) as env:
        with env.begin(write=True) as txn:
            for key, value in lmdb_data.items():
                key = key.encode()
                value = pickle.dumps(value)
                txn.put(key, value)

    print(f"Write lmdb data into {event_path}. ")
    print("The codebook: ", codebook)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing lmdb-format accomplishments. ")
    args = parser.parse_args()
    
    main(args)