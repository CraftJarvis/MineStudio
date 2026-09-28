'''
Date: 2024-11-10 12:22:58
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-10 12:26:49
FilePath: /MineStudio/minestudio/data/minecraft/tools/process_action_openai.py
'''

import os
import json
import pickle
import subprocess
from multiprocessing.pool import ThreadPool, Pool

import av
import cv2
import numpy as np
import argparse
from typing import Dict
from rich import print
from rich.console import Console
from pathlib import Path
from typing import Dict, List, Any, Sequence
from hashlib import md5

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

CAMERA_SCALER = 360.0 / 2400.0

def json_action_to_env_action(json_action):
    """
    Converts a JSON action representation to a MineRL environment action.

    This function takes a JSON object representing player actions (keyboard and mouse)
    and transforms it into a dictionary format compatible with the MineRL environment.
    It handles mapping keyboard keys to specific actions, scaling mouse movements 
    for camera control, and processing mouse button presses.

    :param json_action: A dictionary representing the action in JSON format. 
                        Expected to have "keyboard" and "mouse" keys.
    :type json_action: Dict
    :returns: A tuple containing:
        - env_action: A dictionary representing the action in MineRL format.
        - is_null_action: A boolean indicating if the action is a NOOP (no operation).
    :rtype: Tuple[Dict, bool]
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0., 0.])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action


def merge_action(cache: Dict[str, np.ndarray], action: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Merges a new action into a cache of actions.

    If a key from the new action already exists in the cache, the new action's 
    value is appended to the existing numpy array. If the key is new, it's added
    to the cache with the new action's value as a new numpy array.

    :param cache: The dictionary caching previous actions. Values are numpy arrays.
    :type cache: Dict[str, np.ndarray]
    :param action: The new action dictionary to merge.
    :type action: Dict[str, Any]
    :returns: The updated cache dictionary.
    :rtype: Dict[str, np.ndarray]
    """
    for key, val in action.items():
        if key not in cache:
            cache[key] = np.array([val])
        else:
            cache[key] = np.concatenate((cache[key], np.array([val])), axis=0)
    return cache

    
def run(args_dict: Dict) -> bool:
    """
    Processes a single JSONL file containing actions and converts it to a pickle file.

    This function reads actions from a JSONL file, converts them to the MineRL 
    environment action format, handles potential issues like stuck mouse buttons, 
    and saves the processed actions as a pickle file.

    :param args_dict: A dictionary containing arguments for processing:
        - 'no': The number/identifier of the task.
        - 'name': The base name for the output file.
        - 'json_path': Path to the input JSONL file.
        - 'action_path': Path to save the output pickle file.
    :type args_dict: Dict
    :returns: True if processing was successful, False otherwise.
    :rtype: bool
    """
    no = args_dict['no']
    name = args_dict['name']
    json_path = args_dict['json_path']
    action_path = args_dict['action_path']
    
    print("<No.> {}, name: {}.".format(no, name))

    json_file = json_path.open()
    try:
        json_lines = json_file.readlines()
        json_data = "[" + ",".join(json_lines) + "]"
        json_data = json.loads(json_data)
    except:
        print("<Warning> jsonl file {} is broken, ignored.".format(name))
        return False
    
    nb_json_frames = len(json_data) 

    attack_is_stuck = False
    last_hotbar = 0
    cache_actions = {}
    for i in range(len(json_data)):
        step_data = json_data[i]

        try:
            if i == 0:
                # Check if attack will be stuck down
                if step_data["mouse"]["newButtons"] == [0]:
                    attack_is_stuck = True
            elif attack_is_stuck:
                # Check if we press attack down, then it might not be stuck
                if 0 in step_data["mouse"]["newButtons"]:
                    attack_is_stuck = False
            # If still stuck, remove the action
            if attack_is_stuck:
                step_data["mouse"]["buttons"] = [button for button in step_data["mouse"]["buttons"] if button != 0]

            action, is_null_action = json_action_to_env_action(step_data)
            
            # Update hotbar selection
            current_hotbar = step_data["hotbar"]
            if current_hotbar != last_hotbar:
                action["hotbar.{}".format(current_hotbar + 1)] = 1
            last_hotbar = current_hotbar

            action.pop('ESC')
            action.pop('swapHands')
            action.pop('pickItem')
        except:
            print("Could not process json to action.")
            return False
        
        merge_action(cache_actions, action)

    
    if len(cache_actions) != 21:
        print("<Warning> for {}, the number ({}) is not equal to 21.".format(name, len(cache_actions)))
        return False
    
    '''Save action to pickle file.'''
    with action_path.open('wb') as action_file:
        pickle.dump(cache_actions, action_file)
    
    print("<Success> {}, action frame: {}.".format(name, len(cache_actions['attack'])))
    return True


def main(cfg):
    """
    Main function to process multiple JSONL action files in parallel.

    It scans an input directory for JSONL files, prepares arguments for each file,
    and then uses a multiprocessing Pool to process them in parallel using the `run`
    function. It prints a summary of successful and total processed files.

    :param cfg: Configuration object, expected to have attributes:
        - 'input_dir': The directory containing JSONL files.
        - 'nb_worker': The number of parallel workers to use.
        - 'use_md5': Boolean, if True, generate video ID using MD5 hash of the name.
    :type cfg: argparse.Namespace
    """
    
    root = Path(cfg.input_dir)
    json_dir = root / 'jsonl'
    assert json_dir.is_dir(), "directory (jsonl) must exist!"
    action_dir = root / 'actions'
    
    if not action_dir.is_dir():
        print('Creating directory to save action pickles.')
        action_dir.mkdir(parents=True)
    
    args_list = []
    for no, json_path in enumerate(json_dir.glob('*.jsonl')):
        name = json_path.name.split('.')[-2]
        
        if cfg.use_md5:
            name = md5(name.encode()).hexdigest()[:11]

        action_path = action_dir / (name + '.pkl')
        
        if action_path.exists():
            print("For {}, all files already exist, skip!".format(name))
            continue
        
        args_dict = {
            'no': no,
            'name': name,
            'json_path': json_path,
            'action_path': action_path,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb-worker", help="number of paralle workers", type=int, default=1)
    parser.add_argument("--input-dir", help="working directory", type=str, required=True)
    parser.add_argument("--use-md5", help="use md5 to generate video id", default=False, action='store_true')
    cfg = parser.parse_args()
    print(cfg)
    main(cfg)