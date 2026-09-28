# Slighly modified from human_play_interface.py
# A simple pyglet app which controls the MineRL env,
# showing human the MineRL image and passing game controls
# to MineRL
# Intended for quick data collection without hassle or
# human corrections (agent plays but human can take over).

import datetime

from typing import Callable, Optional, Any
import imgui
from imgui.integrations.pyglet import PygletRenderer
import os
import io
import av
import random
import pickle
import numpy as np
import cv2
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import json
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key
import pyglet.window.mouse as mouse
from omegaconf import DictConfig
from typing import Union, Dict, List
from minestudio.simulator.minerl.env_interface import MinecraftWrapper
# from minestudio.simulator.minerl.entry import env_generator
from omegaconf import OmegaConf

import time

# Mapping from minestudio.simulator.minerl action space names to pyglet keys
MINERL_ACTION_TO_KEYBOARD = {
    #"ESC":       key.ESCAPE, # Used in BASALT to end the episode
    "attack":    mouse.LEFT,
    "back":      key.S,
    #"drop":      key.Q,
    "forward":   key.W,
    "hotbar.1":  key._1,
    "hotbar.2":  key._2,
    "hotbar.3":  key._3,
    "hotbar.4":  key._4,
    "hotbar.5":  key._5,
    "hotbar.6":  key._6,
    "hotbar.7":  key._7,
    "hotbar.8":  key._8,
    "hotbar.9":  key._9,
    "inventory": key.E,
    "jump":      key.SPACE,
    "left":      key.A,
    # "pickItem":  pyglet.window.mouse.MIDDLE,
    "right":     key.D,
    "sneak":     key.LSHIFT,
    "sprint":    key.LCTRL,
    #"swapHands": key.F,
    "use":       mouse.RIGHT, 
    # "switch":    key.TAB,
    # "reset":     key.F1,
}

KEYBOARD_TO_MINERL_ACTION = {v: k for k, v in MINERL_ACTION_TO_KEYBOARD.items()}

IGNORED_ACTIONS = {"chat"}


# Camera actions are in degrees, while mouse movement is in pixels
# Multiply mouse speed by some arbitrary multiplier
MOUSE_MULTIPLIER = 0.1

MINERL_FPS = 25
MINERL_FRAME_TIME = 1 / MINERL_FPS

SCALE = 2
WINDOW_WIDTH = 640*SCALE
WINDOW_HEIGHT = 360*(SCALE+1)

screen = pyglet.canvas.get_display().get_default_screen()
ratio = 0.8

if screen.width < WINDOW_WIDTH * ratio or screen.height < WINDOW_HEIGHT * ratio:
    scale = min(screen.width * ratio / WINDOW_WIDTH, screen.height * ratio / WINDOW_HEIGHT)
    WINDOW_WIDTH = int(WINDOW_WIDTH * scale)
    WINDOW_HEIGHT = int(WINDOW_HEIGHT * scale)
    

FRAME_HEIGHT = WINDOW_HEIGHT // 4 * 3

INFO_WIDTH = WINDOW_WIDTH
INFO_HEIGHT = WINDOW_HEIGHT // 4

NUM_ROWS = 4
NUM_COLS = 6
GRID_WIDTH = WINDOW_WIDTH // NUM_COLS
GRID_HEIGHT = INFO_HEIGHT // NUM_ROWS
GRID = {}
GRID_ID = 0
for R in range(NUM_ROWS):
    for C in range(NUM_COLS):
        X = C * GRID_WIDTH + GRID_WIDTH // 5
        Y = R * GRID_HEIGHT + GRID_HEIGHT // 2
        GRID[GRID_ID] = (X, Y)
        GRID_ID += 1


class RecordHumanPlay(gym.Wrapper):
    def __init__(
        self, 
        env_config: Union[Dict, DictConfig], 
        output_dir: Optional[str] = None, 
        extra_draw_call: Optional[Callable[..., None]] = None,
        **kwargs,
    ) -> None:
        # self.env = env_generator(env_config)[0]
        self.env = MinecraftWrapper(env_config)
        super().__init__(self.env)
        self.start_time = time.time()
        self.end_time = time.time()
        self.env_config = env_config
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise ValueError("Output directory does not exist")
        self.output_dir = output_dir
        self.extra_draw_call = extra_draw_call
        
        self.switch = 'human' # ['human', 'bot']
        self.recording = False
        self.recording_step = 0
        self.timestep = 0
        
        # self._validate_minerl_env(self.env.env)
        # pov_shape = self.env._env.env.observation_space["pov"].shape
        self.window = pyglet.window.Window(
            width=WINDOW_WIDTH,
            height=INFO_HEIGHT + FRAME_HEIGHT,
            vsync=False,
            resizable=False
        )
        imgui.create_context()
        imgui.get_io().display_size = WINDOW_WIDTH, WINDOW_HEIGHT
        self.renderer = PygletRenderer(self.window)
        self.pressed_keys = defaultdict(lambda: False)
        self.released_keys = defaultdict(lambda: False)
        self.window.on_mouse_motion = self._on_mouse_motion
        self.window.on_mouse_drag = self._on_mouse_drag
        self.window.on_key_press = self._on_key_press
        self.window.on_key_release = self._on_key_release
        self.window.on_mouse_press = self._on_mouse_press
        self.window.on_mouse_release = self._on_mouse_release
        self.window.on_activate = self._on_window_activate
        self.window.on_deactive = self._on_window_deactivate
        self.window.dispatch_events()
        self.window.switch_to()
        self.window.flip()

        self.last_pov = None
        self.last_mouse_delta = [0, 0]
        self.capture_mouse = True
        self.chat_message = None

        self.window.clear()
        self._show_message("Waiting for reset.")

    def _on_key_press(self, symbol, modifiers):
        self.pressed_keys[symbol] = True

    def _on_key_release(self, symbol, modifiers):
        self.pressed_keys[symbol] = False
        self.released_keys[symbol] = True

    def _on_mouse_press(self, x, y, button, modifiers):
        self.pressed_keys[button] = True

    def _on_mouse_release(self, x, y, button, modifiers):
        self.pressed_keys[button] = False

    def _on_window_activate(self):
        self.window.set_mouse_visible(False)
        self.window.set_exclusive_mouse(True)

    def _on_window_deactivate(self):
        self.window.set_mouse_visible(True)
        self.window.set_exclusive_mouse(False)

    def _on_mouse_motion(self, x, y, dx, dy):
        # Inverted
        self.last_mouse_delta[0] -= dy * MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * MOUSE_MULTIPLIER

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # Inverted
        self.last_mouse_delta[0] -= dy * MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * MOUSE_MULTIPLIER

    def _validate_minerl_env(self, minerl_env):
        """Make sure we have a valid MineRL environment. Raises if not."""
        # Make sure action has right items
        remaining_buttons = set(MINERL_ACTION_TO_KEYBOARD.keys()).union(IGNORED_ACTIONS)
        remaining_buttons.add("camera")
        for action_name, action_space in minerl_env.action_space.spaces.items():
            if action_name not in remaining_buttons:
                raise RuntimeError(f"Invalid MineRL action space: action {action_name} is not supported.")
            elif not action_name in IGNORED_ACTIONS and (not isinstance(action_space, spaces.Discrete) or action_space.n != 2) and action_name != "camera":
                raise RuntimeError(f"Invalid MineRL action space: action {action_name} had space {action_space}. Only Discrete(2) is supported.")
            remaining_buttons.remove(action_name)
        if len(remaining_buttons) > 0:
            raise RuntimeError(f"Invalid MineRL action space: did not contain actions {remaining_buttons}")

        obs_space = minerl_env.observation_space
        if not isinstance(obs_space, spaces.Dict) or "pov" not in obs_space.spaces:
            raise RuntimeError("Invalid MineRL observation space: observation space must contain POV observation.")
    
    def _update_image(self, arr, message: List = [], **kwargs):
        self.window.switch_to()
        self.window.clear()
        # Based on scaled_image_display.py
        arr = cv2.resize(arr, dsize=(WINDOW_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC) # type: ignore
        image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
        texture = image.get_texture()
        texture.blit(0, INFO_HEIGHT)
        self._show_additional_message(message)
        
        imgui.new_frame()

        if self.extra_draw_call:
            self.extra_draw_call()
        
        imgui.begin("Chat", False, imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        changed, command = imgui.input_text("Message", "")
        if imgui.button("Send"):
            self.chat_message = command
        imgui.end()

        imgui.render()
        self.renderer.render(imgui.get_draw_data())

        self.window.flip()

    def _get_human_action(self):
        """Read keyboard and mouse state for a new action"""
        # Keyboard actions
        action: dict[str, Any] = {
            name: int(self.pressed_keys[key]) for name, key in MINERL_ACTION_TO_KEYBOARD.items()
        }

        if not self.capture_mouse:
            self.last_mouse_delta = [0, 0]
        action["camera"] = self.last_mouse_delta
        self.last_mouse_delta = [0, 0]
        return action

    def _show_message(self, text):
        label = pyglet.text.Label(
            text,
            font_size=32,
            x=self.window.width // 2,
            y=self.window.height // 2,
            anchor_x='center',
            anchor_y='center'
        )
        label.draw()
        self.window.flip()
    
    def _show_additional_message(self, message: List):
        if len(message) == 0:
            return
        line_height = INFO_HEIGHT // len(message)
        y = line_height // 2
        for i, row in enumerate(message):
            line = ' | '.join(row)
            pyglet.text.Label(
                line,
                font_size = 7*SCALE, 
                x = self.window.width // 2, y = y, 
                anchor_x = 'center', anchor_y = 'center',
            ).draw()
            y += line_height

    def _save_trajectory(
        self, 
        dir_name: Optional[str] = None, 
        change_latest=True
    ):
        assert self.output_dir is not None
        if dir_name is None:
            output_dir = Path(self.output_dir)
        else:
            output_dir = Path(self.output_dir) / dir_name
        
        video_dir  = output_dir / "videos"
        action_dir = output_dir / "actions"
        video_dir.mkdir(parents=True, exist_ok=True)
        action_dir.mkdir(parents=True, exist_ok=True)

        info_list, act_list = zip(*self.trajectory)
        outStream = io.BytesIO()
        container = av.open(outStream, mode='w', format='mp4')
        stream = container.add_stream('h264', rate=30)
        stream.width = 640
        stream.height = 360
        stream.pix_fmt = 'yuv420p'
        
        def filter_by_zero(info):
            info = deepcopy(info)
            candidate_set = ['break_item', 'craft_item', 'mine_block', 'kill_entity', 'pickup', 'equipped_items']
            for k, v in info.items():
                if k not in candidate_set:
                    continue
                remove = []
                for i, n in v.items():
                    if n == 0:
                        remove.append(i)
                for i in remove:
                    del v[i]
            return info
        
        save_info_list = []
        for info in info_list:
            img = info.pop("pov")
            save_info_list.append(filter_by_zero(info))
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet) 

        for packet in stream.encode():
            container.mux(packet)
        container.close()

        bytes = outStream.getvalue()
        file_name = datetime.datetime.now().strftime("%H-%M-%S")
        video_path = video_dir / f"{file_name}.mp4"
        with video_path.open("wb") as f:
            f.write(bytes)
        
        action_path = action_dir / f"{file_name}.pkl"
        with action_path.open("wb") as f:
            pkl_acts = {
                key: np.array([act[key] for act in act_list])
                for key in act_list[0].keys()
            }
            pickle.dump(pkl_acts, f)
        
        if change_latest:
            self.latest_video = os.path.abspath(video_path)
        
        # pickle.dump(save_info_list, open(os.path.join(output_dir, f"{file_name}.pkl"), "wb"))


    def reset(self, seed=None):
        if "trajectory" in self.__dict__:
            if self.output_dir is not None:
                self._save_trajectory()
        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randint(0, 2**32 - 1)
        self.env._env.env.seed(self.seed)

        self.terminated = False
        self.window.clear()
        self.pressed_keys = defaultdict(lambda: False)
        self._show_message("Resetting environment...")
        obs, info = self.env.reset()
        self._update_image(info["pov"])
        self.last_info = info
        self.trajectory = []
        self.timestep = 0
        return obs, info
    
    def step(self, action: Optional[Dict[str, Any]] = None, override_if_human_input: bool = False):
        """
        Step environment for one frame.

        If `action` is not None, assume it is a valid action and pass it to the environment.
        Otherwise read action from player (current keyboard/mouse state).

        If `override_if_human_input` is True, execeute action from the human player if they
        press any button or move mouse.

        The executed action will be added to the info dict as "taken_action".
        """
        assert not self.terminated, "Cannot step environment after it is done."
        self.end_time = time.time()
        time.sleep(max(0, MINERL_FRAME_TIME - (self.end_time - self.start_time)))
        fps = 1 / (self.end_time - self.start_time)
        self.start_time = time.time()

        self.window.dispatch_events()
        if not action or override_if_human_input:
            human_action = self._get_human_action()
            if override_if_human_input:
                if any(v != 0 if name != "camera" else (v[0] != 0 or v[1] != 0) for name, v in human_action.items()):
                    action = human_action
            else:
                action = human_action
                
        if self.chat_message is not None:
            action["chat"] = self.chat_message
            self.chat_message = None

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_info = info
        
        self.timestep += 1
        if self.recording:
            self.recording_step += 1
            self.trajectory.append((self.last_info, action))
        
        message = [
            [f"Role: {self.switch}"], 
            [f"Record: {self.recording}", f"Record Step: {self.recording_step}", f"timestep: {self.timestep}", f"FPS: {fps:.2f}"], 
            [f"X: {info['player_pos']['x']:.2f}", f"Y: {info['player_pos']['y']:.2f}", f"Z: {info['player_pos']['z']:.2f}"],
        ]
        
        assert action is not None
        # Display actions
        if len(action) == 2:
            action = MinecraftWrapper.agent_action_to_env(action)
        action_items = []
        for k, v in action.items():
            if k == 'camera':
                v = f"({v[0]:.2f}, {v[1]:.2f})"
            elif 'hotbar' in k:
                continue
            action_items.append(f"{k}: {v}")
        message.append(action_items)
        self._update_image(info["pov"], message=message)

        if self.released_keys[key.C]:
            self.released_keys[key.C] = False
            self.capture_mouse = not self.capture_mouse
            self.window.set_mouse_visible(not self.capture_mouse)
            self.window.set_exclusive_mouse(self.capture_mouse)

        
        if self.released_keys[key.L]:
            self.released_keys[key.L] = False
            self.switch = 'human' if self.switch == 'bot' else 'bot'
        
        if self.released_keys[key.R]:
            self.released_keys[key.R] = False
            # Start & stop recording. 
            if self.recording:
                # save the trajectory
                self._save_trajectory(dir_name='human')
                print('Record Video Path:', self.latest_video)
            self.recording = not self.recording
            self.recording_step = 0
            self.trajectory = []
        
        if self.released_keys[key.F1]:
            self.released_keys[key.F1] = False
            self.env.reset()
        
        if terminated:
            self._show_message("Episode terminated.")
        
        info["taken_action"] = action
        info['switch'] = self.switch
        self.terminated = terminated
        return obs, reward, terminated, truncated, info
