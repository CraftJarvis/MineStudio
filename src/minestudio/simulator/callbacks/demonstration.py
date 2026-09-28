'''
Date: 2025-01-07 05:58:26
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-15 20:40:34
FilePath: /ROCKET-2/var/nfs-shared/shaofei/nfs-workspace/MineStudio/minestudio/simulator/callbacks/demonstration.py
'''
import random
import numpy as np
import os

from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils import get_mine_studio_dir
from minestudio.utils.register import Registers

def download_reference_videos():
    """Downloads reference videos from Hugging Face.

    Retrieves the Minecraft reference videos dataset (CraftJarvis/MinecraftReferenceVideos)
    and saves them to "reference_videos" in the MineStudio root directory.
    """
    import huggingface_hub
    root_dir = get_mine_studio_dir()
    local_dir = os.path.join(root_dir, "reference_videos")
    print(f"Downloading reference videos to {local_dir}")
    huggingface_hub.snapshot_download(repo_id='CraftJarvis/MinecraftReferenceVideos', repo_type='dataset', local_dir=local_dir)

@Registers.simulator_callback.register
class DemonstrationCallback(MinecraftCallback):
    """Provides demonstration data, primarily for GROOT.

    Manages access to task-specific reference videos, including downloading them if absent.

    :param task: Name of the task for demonstration data.
    :type task: str
    """
    
    def create_from_conf(source):
        """Creates a DemonstrationCallback from a configuration.

        Loads data from the source (file path or dict).
        Initializes DemonstrationCallback if 'reference_video' is present.

        :param source: Configuration source.
        :type source: any
        :returns: DemonstrationCallback instance or None.
        :rtype: Optional[DemonstrationCallback]
        """
        data = MinecraftCallback.load_data_from_conf(source)
        if 'reference_video' in data:
            return DemonstrationCallback(data['reference_video'])
        else:
            return None
    
    def __init__(self, task):
        """Initializes DemonstrationCallback.

        Sets up by: identifying reference video directory, prompting for download
        if videos are missing, and selecting a random reference video for the task.

        :param task: The task name.
        :type task: str
        :raises AssertionError: If the task's reference video directory doesn't exist.
        """
        root_dir = get_mine_studio_dir()
        reference_videos_dir = os.path.join(root_dir, "reference_videos")
        if not os.path.exists(reference_videos_dir):
            response = input("Detecting missing reference videos, do you want to download them from huggingface (Y/N)?\n")
            while True:
                if response == 'Y' or response == 'y':
                    download_reference_videos()
                    break
                elif response == 'N' or response == 'n':
                    break
                else:
                    response = input("Please input Y or N:\n")

        self.task = task

        # load the reference video
        ref_video_name = task

        assert os.path.exists(os.path.join(reference_videos_dir, ref_video_name)), f"Reference video {ref_video_name} does not exist."

        ref_video_path = os.path.join(reference_videos_dir, ref_video_name, "human")

        # randomly select a video end with .mp4
        ref_video_list = [f for f in os.listdir(ref_video_path) if f.endswith('.mp4')]

        ref_video_path = os.path.join(ref_video_path, random.choice(ref_video_list))

        self.ref_video_path = ref_video_path

    def after_reset(self, sim, obs, info):
        """Adds the reference video path to the observation dictionary after a reset.

        This method ensures `obs['ref_video_path']` is set with the path to the
        selected demonstration video.

        :param sim: The simulation environment.
        :param obs: The observation dictionary.
        :param info: Additional information dictionary.
        :returns: The modified `obs` and `info`.
        :rtype: tuple[dict, dict]
        """
        obs['ref_video_path'] = self.ref_video_path
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """Adds the reference video path to the observation dictionary after each step.

        This method ensures `obs['ref_video_path']` is set with the path to the
        selected demonstration video.

        :param sim: The simulation environment.
        :param obs: The observation dictionary.
        :param reward: The reward from the current step.
        :param terminated: Whether the episode has terminated.
        :param truncated: Whether the episode has been truncated.
        :param info: Additional information dictionary.
        :returns: The modified `obs`, `reward`, `terminated`, `truncated`, and `info`.
        :rtype: tuple[dict, float, bool, bool, dict]
        """
        obs['ref_video_path'] = self.ref_video_path
        return obs, reward, terminated, truncated, info

    def __repr__(self):
        """Returns a string representation of DemonstrationCallback.

        :returns: String representation.
        :rtype: str
        """
        return f"DemonstrationCallback(task={self.task})"
