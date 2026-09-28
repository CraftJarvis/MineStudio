'''
Date: 2024-11-10 12:24:31
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-10 12:26:26
FilePath: /MineStudio/minestudio/data/minecraft/tools/process_privilege_openai.py
'''

import os
import sys
import json
import pickle
import subprocess
import shutil
import tempfile
from multiprocessing import Pool

import av
import cv2
import numpy as np
import argparse
from typing import Dict
from rich import print
from rich.console import Console
from pathlib import Path
from typing import Dict, List, Any, Sequence, Tuple, Iterator # MODIFIED
from hashlib import md5


def load_tag_items(tag_items_path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Loads item tag information from a JSON file.

    This function reads a JSON file where keys are tag groups (e.g., "minecraft:logs")
    and values are lists of items belonging to that group (e.g., ["minecraft:oak_log", "minecraft:birch_log"]).
    It processes this data to create two dictionaries:
    1.  `group_to_items`: Maps a group name (e.g., "logs") to a list of item names (e.g., ["oak_log", "birch_log"]).
    2.  `item_to_group`: Maps an item name (e.g., "oak_log") to its corresponding group name (e.g., "logs").

    :param tag_items_path: The file path to the JSON file containing tag items.
    :type tag_items_path: str
    :returns: A tuple containing two dictionaries:
              - group_to_items (Dict[str, List[str]]): Maps group names to lists of item names.
              - item_to_group (Dict[str, str]): Maps item names to their group names.
    :rtype: Tuple[Dict[str, List[str]], Dict[str, str]]
    """
    
    with open(tag_items_path, 'r') as f:
        tag_items = json.load(f)
    item_to_group = {}
    group_to_items = {}
    for group, val in tag_items.items():
        group_name = group.split(':')[1]
        if group_name not in group_to_items:
            group_to_items[group_name] = []
        for item in val:
            item_name = item.split(':')[1]
            item_to_group[item_name] = group_name
            group_to_items[group_name].append(item_name)
            
    return group_to_items, item_to_group


def parse_crafting_items(recipe_path: str) -> Dict[str, Dict]:
    """
    Parses Minecraft recipe files to create a summary of crafting ingredients and results.

    It iterates through JSON recipe files in the specified path, focusing on
    'minecraft:crafting_shaped', 'minecraft:crafting_shapeless', and 'minecraft:smelting' types.
    For each valid recipe, it extracts the required items/tags and the resulting item(s) with their counts.
    The summary is stored in a dictionary where keys are the names of the resulting items.

    :param recipe_path: The directory path containing Minecraft JSON recipe files.
    :type recipe_path: str
    :returns: A dictionary where keys are the names of craftable items and values are
              dictionaries summarizing their recipes (ingredients by item/group and results).
    :rtype: Dict[str, Dict]
    """

    crafting_summary = {}
    
    for recipe in Path(RECIPES_PATH).iterdir():
        
        with recipe.open() as f:
            recipe_data = json.load(f)
        
        if recipe_data['type'] not in ['minecraft:crafting_shaped', 'minecraft:crafting_shapeless', 'minecraft:smelting']:
            continue
        
        summary = {
            'item': [],
            'group': [],
            'result': [],
        }
        
        if recipe_data['type'] == 'minecraft:crafting_shaped':
            key_count = {}
            for p in recipe_data['pattern']:
                for c in p:
                    if c != ' ':
                        key_count[c] = key_count.get(c, 0) + 1
            for key, val in recipe_data['key'].items():
                if 'item' in val:
                    _item_name = val['item'].split(':')[1]
                    summary['item'].append((_item_name, key_count[key]))
                if 'tag' in val:
                    _group_name = val['tag'].split(':')[1]
                    summary['group'].append((_group_name, key_count[key]))
            
            item_name = recipe_data['result']['item'].split(':')[1]
            if 'count' in recipe_data['result']:
                summary['result'].append((item_name, recipe_data['result']['count']))
            else:
                summary['result'].append((item_name, 1))
        
        elif recipe_data['type'] == 'minecraft:crafting_shapeless':
            for ingr in recipe_data['ingredients']:
                if 'item' in ingr:
                    _item_name = ingr['item'].split(':')[1]
                    summary['item'].append((_item_name, 1))
                if 'tag' in ingr:
                    _group_name = ingr['tag'].split(':')[1]
                    summary['group'].append((_group_name, 1))
                        
            item_name = recipe_data['result']['item'].split(':')[1]
            if 'count' in recipe_data['result']:
                summary['result'].append((item_name, recipe_data['result']['count']))
            else:
                summary['result'].append((item_name, 1))
        
        elif recipe_data['type'] == 'minecraft:smelting':
            
            ingr = recipe_data['ingredient']
            if 'item' in ingr:
                _item_name = ingr['item'].split(':')[1]
                summary['item'].append((_item_name, 1))
            if 'tag' in ingr:
                _group_name = ingr['tag'].split(':')[1]
                summary['group'].append((_group_name, 1))

            item_name = recipe_data['result'].split(':')[1]
            if 'count' in recipe_data:
                summary['result'].append((item_name, recipe_data['count']))
            else:
                summary['result'].append((item_name, 1))
        
        crafting_summary[item_name] = summary
        
    return crafting_summary

PROJECT_ABSOLUTE_PATH = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )
    )
)

RECIPES_PATH = os.path.join(PROJECT_ABSOLUTE_PATH, 'assets/recipes')
TAG_ITEMS_PATH = os.path.join(PROJECT_ABSOLUTE_PATH, 'assets/tag_items.json')
CRAFTING_SUMMARY = parse_crafting_items(RECIPES_PATH)
GROUP_TO_ITEMS, ITEM_TO_GROUP = load_tag_items(TAG_ITEMS_PATH)

FRAME_SHAPE0 = 360
MINEREC_ORIGINAL_HEIGHT_PX = 720

def get_crafting_items(curr_stats: Dict, prev_stats: Dict) -> Iterator[Tuple[str, int]]: # MODIFIED
    for item in curr_stats:
        if 'craft_item' not in item:
            continue
        item_name = item.split('.')[-1]
        delta = curr_stats.get(item, 0) - prev_stats.get(item, 0)
        if delta > 0:
            yield ('craft_item:' + item_name, delta)

def get_mining_blocks(curr_stats: Dict, prev_stats: Dict) -> Iterator[Tuple[str, int]]: # MODIFIED
    for block in curr_stats:
        if 'mine_block' not in block:
            continue
        block_name = block.split('.')[-1]
        delta = curr_stats.get(block, 0) - prev_stats.get(block, 0)
        if delta > 0:
            yield ('mine_block:' + block_name, delta)

def get_picking_items(curr_stats: Dict, prev_stats: Dict) -> Iterator[Tuple[str, int]]: # MODIFIED
    for item in curr_stats:
        if 'pickup' not in item:
            continue
        item_name = item.split('.')[-1]
        delta = curr_stats.get(item, 0) - prev_stats.get(item, 0)
        if delta > 0:
            yield ('pickup:' + item_name, delta)

def get_other_items(curr_inventory: Dict, prev_inventory: Dict) -> Iterator[Tuple[str, str, int]]: # MODIFIED
    for item_name in curr_inventory:
        delta = curr_inventory.get(item_name, 0) - prev_inventory.get(item_name, 0)
        if delta > 0:
            yield ('inventory:', item_name, delta)


def extract_privileged_info(curr_data: Dict, prev_data: Dict=None) -> Dict:
    
    if prev_data is None:
        prev_data = {}
    
    result = {}
    result['yaw'] = curr_data['yaw']
    result['pitch'] = curr_data['pitch']
    result['xpos'] = curr_data['xpos']
    result['ypos'] = curr_data['ypos']
    result['zpos'] = curr_data['zpos']
    result['hotbar'] = curr_data['hotbar']
    result['inventory'] = curr_data['inventory']
    result['isGuiOpen'] = curr_data['isGuiOpen']
    result['isGuiInventory'] = curr_data['isGuiInventory']
    
    result['delta_yaw'] = curr_data['yaw'] - prev_data.get('yaw', 0)
    result['delta_pitch'] = curr_data['pitch'] - prev_data.get('pitch', 0)
    result['delta_craft_item'] = {}
    result['delta_mine_block'] = {}
    result['delta_pickup'] = {}
    
    if curr_data['isGuiOpen']:
        camera_scaling_factor = FRAME_SHAPE0 / MINEREC_ORIGINAL_HEIGHT_PX
        cursor_x = int(curr_data["mouse"]["x"] * camera_scaling_factor)
        cursor_y = int(curr_data["mouse"]["y"] * camera_scaling_factor)
        result['cursor_x'] = cursor_x
        result['cursor_y'] = cursor_y
    else:
        result['cursor_x'] = 0
        result['cursor_y'] = 0
    
    if 'stats' in prev_data:
        curr_stats = curr_data['stats']
        prev_stats = prev_data['stats']
        
        for k, v in get_crafting_items(curr_stats, prev_stats):
            result['delta_craft_item'][k] = v
        
        for k, v in get_mining_blocks(curr_stats, prev_stats):
            result['delta_mine_block'][k] = v
        
        for k, v in get_picking_items(curr_stats, prev_stats):
            result['delta_pickup'][k] = v
        
    return result

def attach_predicted_results(info: Dict, curr_data: Dict) -> Dict:
    info['pred_recog'] = curr_data


def run(args_dict: Dict) -> bool:
    no = args_dict['no']
    name = args_dict['name']
    json_path = args_dict['json_path']
    privilege_path = args_dict['privilege_path']
    recog_path = args_dict['recog_path']
    
    # print("<No. {}> {}.".format(no, name))
    json_file = json_path.open()
    try:
        json_lines = json_file.readlines()
        json_data = "[" + ",".join(json_lines) + "]"
        json_data = json.loads(json_data)
    except:
        print("<Warning> jsonl file {} is broken, ignored.".format(name))
        return False
    
    if recog_path is not None:
        with recog_path.open() as recog_file:
            recog_data = json.load(recog_file)
    else:
        recog_data = None
    
    privileged_info = []
    prev_data = None

    for i in range(len(json_data)):
        step_data = json_data[i]

        try:
            step_privileged_info = extract_privileged_info(step_data, prev_data)
        except:
            print(f"Could not extract privileged information from {json_path}")
            return False
        
        if recog_data is not None:
            try:
                attach_predicted_results(step_privileged_info, recog_data[i])
            except:
                print(f"Could not extract recog information from {recog_path}")
                return False
        
        privileged_info.append(step_privileged_info)
        prev_data = step_data
        
    '''Save previleged info to pickle file.'''
    with privilege_path.open('wb') as privilege_file:
        pickle.dump(privileged_info, privilege_file)
    
    print("<Success> {}, privilege frame: {}.".format(name, len(privileged_info)))
    return True


def main(cfg):
    
    root = Path(cfg.input_dir)
    json_dir = root / 'jsonl'
    recog_dir = root / 'recog'
    assert json_dir.is_dir(), "directory jsonl must exist!"
    if not os.path.exists(recog_dir):
        recog_dir = None
    
    privilege_dir = root / 'privileged_infos'
    
    
    if not privilege_dir.is_dir():
        print('Creating directory to save privileged information.')
        privilege_dir.mkdir(parents=True)
    
    args_list = []
    for no, json_path in enumerate(json_dir.glob('*.jsonl')):
        # name = json_path.name.split('.')[-2]
        name = json_path.stem
        if recog_dir is not None:
            recog_path = recog_dir / (name + '.json')
            if not recog_path.exists():
                print("For {}, recog file does not exist, skip!".format(name))
                continue
        else:
            recog_path = None
        
        if cfg.use_md5:
            name = md5(name.encode()).hexdigest()[:11]
        
        privilege_path = privilege_dir / (name + '.pkl')
        
        if privilege_path.exists():
            print("For {}, all files already exist, skip!".format(name))
            continue
        
        args_dict = {
            'no': no,
            'name': name,
            'json_path': json_path,
            'privilege_path': privilege_path, 
            'recog_path': recog_path, 
        }
        args_list.append(args_dict)
    
    with Pool(cfg.nb_worker) as pool:
        succ = 0
        total = 0
        for f in pool.map(run, args_list):
            if f:
                succ += 1
            total += 1
    print("<FINISH> successful number: {}, success rate: {:.2f}".format(succ, succ / total))
    

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb-worker", help="number of paralle workers", type=int, default=1)
    parser.add_argument("--input-dir", help="working directory", type=str, required=True)
    parser.add_argument("--use-md5", help="use md5 to generate video id", default=False, action='store_true')
    cfg = parser.parse_args()
    print(cfg)
    main(cfg)