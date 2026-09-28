'''
Date: 2025-05-20 18:18:38
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-05-20 18:23:37
FilePath: /MineStudio/minestudio/online/utils/train/training_session.py
'''
from numpy import roll
from omegaconf import OmegaConf
from omegaconf import DictConfig
import ray
import wandb
import uuid
import torch

@ray.remote(resources={"wandb": 1})
class TrainingSession:
    """
    A Ray remote actor for managing a training session, primarily for logging with Weights & Biases (wandb).

    This class initializes a wandb run and provides methods to log data, define metrics, and log videos.
    It ensures that wandb initialization and logging happen in a dedicated Ray actor.

    :param logger_config: Configuration for the wandb logger, passed directly to `wandb.init()`.
    :param hyperparams: Dictionary of hyperparameters to log with wandb.
    """
    def __init__(self, logger_config: DictConfig, hyperparams: DictConfig):
        self.session_id = str(uuid.uuid4())
        hyperparams_dict = OmegaConf.to_container(hyperparams, resolve=True)
        wandb.init(config=hyperparams_dict, **logger_config) # type: ignore
    
    def log(self, *args, **kwargs):
        """
        Logs data to the current wandb run.

        :param args: Positional arguments passed directly to `wandb.log()`.
        :param kwargs: Keyword arguments passed directly to `wandb.log()`.
        """
        wandb.log(*args, **kwargs)
    
    def define_metric(self, *args, **kwargs):
        """
        Defines a metric for the current wandb run.

        :param args: Positional arguments passed directly to `wandb.define_metric()`.
        :param kwargs: Keyword arguments passed directly to `wandb.define_metric()`.
        """
        wandb.define_metric(*args, **kwargs)
    
    def log_video(self, data: dict, video_key: str, fps: int):
        """
        Logs a video to the current wandb run.

        The video data in the `data` dictionary under `video_key` is converted to a `wandb.Video` object.

        :param data: A dictionary containing the data to log. The video itself should be under the `video_key`.
        :param video_key: The key in the `data` dictionary that holds the video data.
        :param fps: The frames per second of the video.
        """
        data[video_key] = wandb.Video(data[video_key], fps=fps, format="mp4")
        wandb.log(data)

    def get_session_id(self):
        """
        Returns the unique ID of this training session.

        :returns: The session ID string.
        """
        return self.session_id