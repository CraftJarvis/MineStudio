from multiprocessing.connection import Connection
from torch.multiprocessing import Process
from typing import Dict, Callable, Optional, Tuple, Union
import cv2
import torch, rich
import traceback
import av
import time
# import tracemalloc
from omegaconf import DictConfig
import uuid
from threading import Thread
from queue import Queue
import numpy as np
from pathlib import Path, PosixPath
import random
import logging
import os
#from minestudio.simulator.entry import MinecraftSim
import copy
from minestudio.simulator import MinecraftSim
import subprocess
import io
from PIL import Image
import matplotlib.pyplot as plt

class VideoWriter(Thread):
    """
    A class for writing video frames to a file in a separate thread.

    :param video_fps: The frames per second of the output video.
    :param queue_size: The maximum size of the command queue.
    """
    def __init__(self, video_fps: int, queue_size: int = 200):
        super().__init__()
        self.cmd_queue = Queue(queue_size)
        self.video_container = None
        self.video_stream = None
        self.video_fps = video_fps
        self.openess = 0
    def open_video(self, path: PosixPath):
        """
        Opens a video file for writing.

        :param path: The path to the video file.
        :raises AssertionError: if a video is already open.
        """
        assert self.openess == 0
        self.cmd_queue.put(("open", path))
        self.openess = 1
        
    def close_video(self):
        """
        Closes the current video file.
        """
        self.cmd_queue.put(("close",))
        self.openess = 0
        
    def write_frame(self, frame: np.ndarray):
        """
        Writes a frame to the video file.

        :param frame: The frame to write, as a NumPy array.
        :raises AssertionError: if no video is open.
        """
        assert self.openess == 1
        self.cmd_queue.put(("write", frame))
    
    def run(self):
        """
        The main loop of the video writer thread.
        Processes commands from the queue to open, write, and close video files.
        """
        while True:
            cmd, *args = self.cmd_queue.get()
            if cmd == "open":
                path, = args
                if not path.parent.exists():
                    try:
                        path.parent.mkdir(parents=True)
                    except FileExistsError:
                        pass
                assert self.video_container is None
                self._next_path = path # lazy open
            elif cmd == "write":
                if self.video_container is None:
                    self.video_container = av.open(str(self._next_path), mode='w', format='mp4')
                    self.video_stream = self.video_container.add_stream('h264', rate=self.video_fps)
                    height, width, _ = args[0].shape
                    self.video_stream.width = width
                    self.video_stream.height = height
                    self.video_stream.pix_fmt = 'yuv420p'
                frame = av.VideoFrame.from_ndarray(args[0], format='rgb24')
                for packet in self.video_stream.encode(frame):
                    self.video_container.mux(packet)
            elif cmd == "close":
                assert self.video_container is not None
                for packet in self.video_stream.encode():
                    self.video_container.mux(packet)
                self.video_stream.close()
                self.video_container.close()
                self.video_container = None
                self.video_stream = None

def draw_vpred(
    img: np.ndarray,
    vpred: float,
    additional_text: Optional[str] = "",
    text_color: Tuple[int, int, int] = (255, 255, 85),  # Minecraft yellow
    shadow_color: Tuple[int, int, int] = (0, 0, 0),     # Black shadow
) -> np.ndarray:
    """
    Draw vpred text with Minecraft-styled rendering, coherent with before_render function in base_callback2.
    Uses matplotlib for high-quality text rendering with shadow effect.
    """
    # Ensure image is in RGB format
    draw_img = img.copy()
    if len(draw_img.shape) == 2 or draw_img.shape[2] == 1:
        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)

    image_height, image_width, _ = draw_img.shape
    
    # Prepare text content
    text = f"vpred: %0.3f" % vpred + additional_text

    # --- Matplotlib Figure Setup for High DPI (matching base_callback2) ---
    # Set a high DPI for a crisp image
    dpi = 300
    fig = plt.figure(figsize=(image_width / dpi, image_height / dpi), dpi=dpi, facecolor='none', edgecolor='none')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(draw_img)
    ax.axis('off')

    # --- Add Minecraft-Styled Text (matching base_callback2) ---
    # Colors inspired by Minecraft's UI (convert RGB to matplotlib format)
    mc_yellow = f'#{text_color[0]:02x}{text_color[1]:02x}{text_color[2]:02x}'
    mc_black = f'#{shadow_color[0]:02x}{shadow_color[1]:02x}{shadow_color[2]:02x}'
    
    # Text properties (matching base_callback2 style)
    text_props = {
        'transform': ax.transAxes,
        'fontsize': 4,
        'verticalalignment': 'center',
        'horizontalalignment': 'left',
        'fontweight': 'bold',  # Bold to mimic pixel font thickness
    }

    # Position text at left center (similar to base_callback2 but different position)
    text_x = 0.02
    text_y = 0.5

    # Draw the black "shadow" text slightly offset
    # This creates a classic 8-bit style outline/shadow
    shadow_offset = 1 / dpi  # Small offset in points
    ax.text(text_x + shadow_offset, text_y - shadow_offset, text, color=mc_black, **text_props)
    
    # Draw the main yellow text on top
    ax.text(text_x, text_y, text, color=mc_yellow, **text_props)

    # --- Convert Matplotlib Figure back to NumPy Array (matching base_callback2) ---
    with io.BytesIO() as buff:
        fig.savefig(buff, format='png', dpi=dpi, transparent=True, pad_inches=0)
        buff.seek(0)
        # Use PIL to correctly handle the RGBA conversion from PNG
        plot_pil = Image.open(buff)
        plot_image = np.array(plot_pil)

    plt.close(fig)  # Important to free up memory

    # Return the image with the alpha channel sliced off (RGB)
    return plot_image[:, :, :3]

class EnvWorker(Process):
    """
    A class for running a Minecraft simulation environment in a separate process.

    :param env_generator: A function that returns a MinecraftSim instance.
    :param conn: A multiprocessing connection object for communication with the main process.
    :param video_output_dir: The directory to save output videos to.
    :param video_fps: The frames per second for output videos.
    :param restart_interval: The interval in seconds after which to restart the environment.
    :param max_fast_reset: The maximum number of fast resets to perform.
    :param env_id: The ID of the environment.
    :param rollout_worker_id: The ID of the rollout worker.
    """
    def __init__(self, env_generator: Callable[[], MinecraftSim], conn: Connection, video_output_dir: str, video_fps: int, restart_interval: Optional[int] = None, max_fast_reset: int = 10000, env_id: int = 0, rollout_worker_id: int = 0):
        super().__init__()
        self.max_fast_reset = max_fast_reset
        self.env_generator = copy.deepcopy(env_generator)
        self.env_id = env_id
        self.conn = conn
        self.restart_interval = restart_interval
        self.video_output_dir = Path(video_output_dir)
        self.rollout_worker_id = rollout_worker_id
        if not self.video_output_dir.exists():
            try:
                self.video_output_dir.mkdir(parents=True)
            except FileExistsError:
                pass
        
        self.video_fps = video_fps
    
    def step_agent(self, obs: dict, last_reward: float, last_terminated: bool, last_truncated: bool, episode_uuid: str) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Sends an observation to the main process and receives an action and predicted value.

        :param obs: The current observation from the environment.
        :param last_reward: The reward from the previous step.
        :param last_terminated: Whether the previous episode terminated.
        :param last_truncated: Whether the previous episode was truncated.
        :param episode_uuid: The UUID of the current episode.
        :returns: A tuple containing the action and predicted value.
        """
        self.conn.send(("step_agent", obs, last_reward, last_terminated, last_truncated, episode_uuid))
        action, vpred = self.conn.recv()
        return action, vpred

    def reset_state(self) -> Dict[str, torch.Tensor]:
        """
        Sends a reset signal to the main process and receives the initial observation.

        :returns: The initial observation from the environment.
        """
        self.conn.send(("reset_state", None))
        return self.conn.recv()
    
    def report_rewards(self, rewards: np.ndarray, task: Optional[str] = None):
        """
        Sends the rewards for an episode to the main process.

        :param rewards: A NumPy array of rewards for the episode.
        :param task: An optional string specifying the task configuration.
        :returns: The result from the main process.
        """
        self.conn.send(("report_rewards", rewards, task))
        return self.conn.recv()

    def run(self) -> None:
        """
        The main loop of the environment worker process.
        Handles environment resets, steps, and video recording.
        """
        video_writer = VideoWriter(video_fps=self.video_fps)
        video_writer.start()
        record = False
        self.env = self.env_generator()
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        episode_info = None
        while True:
            time.sleep(random.randint(0,20))
            self.env.close()
            self.env = self.env_generator()
            try:
                start_time = time.time()
                reward = 0.0
                terminated, truncated = True, False # HACK: make sure the 'first' flag of the first step in the first episode is True
                while True:
                    if self.restart_interval is not None and time.time() - start_time >= self.restart_interval:
                        raise Exception("Restart interval reached")
                    self.reset_state()
                    obs, info = self.env.reset()
                    reward_list = []
                    step = 0
                    v_preds = []
                    obs_imgs = []
                    episode_uuid = str(uuid.uuid4())
                    while True:
                        if step%100==1:
                            logging.getLogger("ray").info(f"working..., max_fast_reset: {self.max_fast_reset}, env_id: {self.env_id}, rollout_worker_id: {self.rollout_worker_id}, step: {step}")
                        step += 1
                        action, vpred = self.step_agent(obs, 
                                            last_reward=float(reward),
                                            last_terminated=terminated,
                                            last_truncated=truncated,
                                            episode_uuid=episode_uuid
                                        )

                        v_preds.append(vpred)
                        if record:
                            render_image = self.env.render()
                            obs_imgs.append(torch.from_numpy(render_image).unsqueeze(0))
                        obs, reward, terminated, truncated, info = self.env.step(action)
                
                        reward_list.append(reward)
                        if terminated or truncated:
                            break

                    if record:
                        record = False
                        obs_img = torch.cat(obs_imgs, dim=0).numpy()
                        imgs = obs_img.astype(np.uint8)
                        for jdx in range(imgs.shape[0]):
                            img = draw_vpred(imgs[jdx], v_preds[jdx], additional_text="")
                            video_writer.write_frame(img)
                        video_writer.close_video()
                    #_result = self.report_rewards(np.array(reward_list))
                    
                    _result, episode_info = self.report_rewards(np.array(reward_list), obs.get("task", None))
                    obs["online_info"] = episode_info
                    
                    if _result is not None:
                        record = True
                        video_step = _result
                        vidoe_uuid = str(uuid.uuid4())
                        
                        save_video_name = f"{timestamp}/"+f"{video_step} - {vidoe_uuid}.mp4".replace('/', '_').replace('\\', '_')
                        video_path = self.video_output_dir / save_video_name
                        os.makedirs(video_path.parent, exist_ok=True)
                        if video_writer.openess == 1:
                            video_writer.close_video()
                        video_writer.open_video(video_path)

            except Exception as e:
                traceback.print_exc()
                rich.print(f"[bold red]An error occurred in EnvWorker: {e}[/bold red]")