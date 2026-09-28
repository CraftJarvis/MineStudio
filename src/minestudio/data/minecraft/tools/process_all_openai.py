'''
Date: 2024-11-10 12:24:45
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-10 12:26:41
FilePath: /MineStudio/minestudio/data/minecraft/tools/process_all_openai.py
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

MINEREC_ORIGINAL_HEIGHT_PX = 720

# If GUI is open, mouse dx/dy need also be adjusted with these scalers.
# If data version is not present, assume it is 1.
MINEREC_VERSION_SPECIFIC_SCALERS = {
    "5.7": 0.5, 
    "5.8": 0.5, 
    "6.7": 2.0, 
    "6.8": 2.0, 
    "6.9": 2.0, 
}

CURSOR_FILE = os.path.join(os.path.dirname(__file__), "cursors", "mouse_cursor_white_16x16.png")

cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
# Assume 16x16
cursor_image = cursor_image[:16, :16, :]
cursor_alpha = cursor_image[:, :, 3:] / 255.0
cursor_image = cursor_image[:, :, :3]

def json_action_to_env_action(json_action):
    """
    Converts a json action from the input data into a MineRL environment-compatible action.

    It translates keyboard inputs and mouse movements/buttons from the JSON format
    to the action format expected by the MineRL environment.

    :param json_action: A dictionary representing the action from the JSON data,
                        containing keyboard and mouse inputs.
    :type json_action: dict
    :returns: A tuple containing:
              - env_action (dict): The action formatted for the MineRL environment.
              - is_null_action (bool): True if the action is a "no-op" (no significant input), False otherwise.
    :rtype: tuple
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

def composite_images_with_alpha(image1, image2, alpha, x, y):
    """
    Composites image2 onto image1 at a specified (x, y) location using an alpha channel for opacity.

    The function modifies image1 in-place.

    :param image1: The base image (NumPy array) to which image2 will be added.
    :type image1: numpy.ndarray
    :param image2: The image to overlay on top of image1 (NumPy array).
    :type image2: numpy.ndarray
    :param alpha: The alpha channel (NumPy array) for image2, determining its opacity.
    :type alpha: numpy.ndarray
    :param x: The x-coordinate on image1 where the top-left corner of image2 will be placed.
    :type x: int
    :param y: The y-coordinate on image1 where the top-left corner of image2 will be placed.
    :type y: int
    """
    # Modifies image1 in-place
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0:  
        return
    alpha = alpha[:ch, :cw]
    image1[y:y + ch, x:x + cw, :] = (image1[y:y + ch, x:x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha).astype(np.uint8)


def merge_action(cache: Dict[str, np.ndarray], action: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Merges a new action dictionary into a cache of actions.

    If a key from the new action already exists in the cache, the new action's value
    (converted to a NumPy array if it isn't already) is concatenated to the existing
    NumPy array. If the key doesn't exist, it's added to the cache with the new
    action's value as a new NumPy array.

    :param cache: The dictionary caching previous actions. Keys are action types (str),
                  and values are NumPy arrays of action values.
    :type cache: Dict[str, np.ndarray]
    :param action: The new action dictionary to merge. Keys are action types (str),
                   and values can be of any type that can be converted to a NumPy array.
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


def write_video(
    file_name: str, 
    frames: Sequence[np.ndarray], 
    width: int = 640, 
    height: int = 360, 
    fps: int = 20
) -> None:
    """Write video frames to a video file using the PyAV library.

    This function takes a sequence of NumPy array frames and encodes them into a video file
    with the specified properties (filename, width, height, FPS).

    :param file_name: The name (including path) of the output video file.
    :type file_name: str
    :param frames: A sequence of frames, where each frame is a NumPy array (RGB format).
    :type frames: Sequence[np.ndarray]
    :param width: The width of the output video in pixels, defaults to 640.
    :type width: int, optional
    :param height: The height of the output video in pixels, defaults to 360.
    :type height: int, optional
    :param fps: The frames per second of the output video, defaults to 20.
    :type fps: int, optional
    """
    """Write video frames to video files. """
    with av.open(file_name, mode="w", format='mp4') as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

def extract_privileged_info(curr_data: Dict, prev_data: Dict=None) -> Dict:
    """
    Extracts privileged information from the current game state data and calculates deltas from previous state.

    Privileged information includes player's yaw, pitch, position (x, y, z), hotbar contents,
    and full inventory. It also calculates the change in yaw, pitch, and inventory items
    compared to the previous game state.

    :param curr_data: A dictionary containing the current game state data.
    :type curr_data: Dict
    :param prev_data: A dictionary containing the previous game state data, defaults to None.
                      If None, delta values will be calculated against zeros or empty inventories.
    :type prev_data: Dict, optional
    :returns: A dictionary containing the extracted privileged information and calculated deltas.
    :rtype: Dict
    """
    
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
    
    result['delta_yaw'] = curr_data['yaw'] - prev_data.get('yaw', 0)
    result['delta_pitch'] = curr_data['pitch'] - prev_data.get('pitch', 0)
    result['delta_inventory'] = {}
    
    if 'inventory' in prev_data:
        curr_inventory = {}
        assert 'inventory' in curr_data, "Current data must have inventory."
        for item in curr_data['inventory']:
            curr_inventory[item['type']] = curr_inventory.get(item['type'], 0) + item['quantity']

        prev_inventory = {}
        for item in prev_data['inventory']:
            prev_inventory[item['type']] = prev_inventory.get(item['type'], 0) + item['quantity']
        
        for key in curr_inventory.keys():
            delta = curr_inventory.get(key, 0) - prev_inventory.get(key, 0)
            if delta > 0:
                result['delta_inventory'][key] = delta
                # print(key, delta, curr_inventory, prev_inventory)
        
    return result
    
def run(args_dict: Dict) -> bool:
    
    name = args_dict['name']
    mp4_path = args_dict['mp4_path']
    json_path = args_dict['json_path']
    video_path = args_dict['video_path']
    action_path = args_dict['action_path']
    privilege_path = args_dict['privilege_path']
    
    print("<Start> {}.".format(name))
    video = cv2.VideoCapture(str(mp4_path))
    nb_mp4_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS)) 
    
    if fps != 20:
        print("<Warning> fps {} is not equal to 20, ignored.".format(fps))
        return False
    
    json_file = json_path.open()
    try:
        json_lines = json_file.readlines()
        json_data = "[" + ",".join(json_lines) + "]"
        json_data = json.loads(json_data)
    except:
        print("<Warning> jsonl file {} is broken, ignored.".format(name))
        return False
    
    nb_json_frames = len(json_data) 
    if not (nb_json_frames <= nb_mp4_frames <= nb_json_frames + 1):
        print("<Warning> frame number of mp4 ({}) and json ({}) is incosistent.".format(nb_mp4_frames, nb_json_frames))
        return False

    attack_is_stuck = False
    last_hotbar = 0
    cache_actions = {}
    output_frames = []
    privileged_info = []
    prev_data = None
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
        
        ret, frame = video.read()
        if ret:
            if step_data["isGuiOpen"]:
                try:
                    camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
                    cursor_x = int(step_data["mouse"]["x"] * camera_scaling_factor)
                    cursor_y = int(step_data["mouse"]["y"] * camera_scaling_factor)
                    composite_images_with_alpha(frame, cursor_image, cursor_alpha, cursor_x, cursor_y)
                except:
                    print(f"Could not add cursor!")
                    return False
            else:
                cursor_x = 0
                cursor_y = 0
            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
            output_frames.append(frame)
        else:
            print(f"Could not read frame from video {mp4_path}")
            return False
        
        try:
            step_privileged_info = extract_privileged_info(step_data, prev_data)
            step_privileged_info['cursor_x'] = cursor_x
            step_privileged_info['cursor_y'] = cursor_y
            privileged_info.append(step_privileged_info)
        except:
            print(f"Could not extract privileged information from {privilege_path}")
            return False
        
        prev_data = step_data
    video.release()
    
    if len(cache_actions) != 21:
        print("<Warning> for {}, the number ({}) is not equal to 21.".format(name, len(cache_actions)))
    
    '''Save action to pickle file.'''
    with action_path.open('wb') as action_file:
        pickle.dump(cache_actions, action_file)
    
    '''Save previleged info to pickle file.'''
    with privilege_path.open('wb') as privilege_file:
        pickle.dump(privileged_info, privilege_file)
    
    '''Save video to mp4 file.'''
    write_video(str(video_path), output_frames, width, height, fps)
    
    print("<Success> {}, video frame: {}, action frame: {}.".format(name, len(output_frames), len(cache_actions['attack'])))
    return True


def main(cfg):
    
    root = Path(cfg.input_dir)
    mp4_dir = root / 'mp4'
    json_dir = root / 'jsonl'
    assert mp4_dir.is_dir() and json_dir.is_dir(), "directory (mp4 and jsonl) must exist!"
    action_dir = root / 'actions'
    video_dir = root / 'videos'
    privilege_dir = root / 'privileged_infos'
    
    if not action_dir.is_dir():
        print('Creating directory to save action pickles.')
        action_dir.mkdir(parents=True)
    if not video_dir.is_dir():
        print('Creating directory to save videos.')
        video_dir.mkdir(parents=True)
    if not privilege_dir.is_dir():
        print('Creating directory to save privileged information.')
        privilege_dir.mkdir(parents=True)
    
    
    args_list = []
    for json_path in json_dir.glob('*.jsonl'):
        name = json_path.name.split('.')[-2]
        mp4_path = mp4_dir / (name + '.mp4')
        if not mp4_path.exists():
            print("For {}, only jsonl exists, mp4 file does not exist!".format(name))
            continue
        
        if cfg.use_md5:
            name = md5(name.encode()).hexdigest()[:11]
            # action_path = md5(action_path.encode()).hexdigest()[:11]
            # video_path = md5(video_path.encode()).hexdigest()[:11]
            # privilege_path = md5(video_path.encode()).hexdigest()[:11]
        
        action_path = action_dir / (name + '.pkl')
        video_path = video_dir / (name + '.mp4')
        privilege_path = privilege_dir / (name + '.pkl')
        
        if action_path.exists() and video_path.exists() and privilege_path.exists():
            print("For {}, all files already exist, skip!".format(name))
            continue
        
        args_dict = {
            'name': name,
            'json_path': json_path,
            'mp4_path': mp4_path,
            'action_path': action_path,
            'video_path': video_path, 
            'privilege_path': privilege_path,
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