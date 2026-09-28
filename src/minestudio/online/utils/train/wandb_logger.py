from typing import Dict, Any
import ray
import logging
from minestudio.online.utils.train import get_current_session

logger = logging.getLogger("ray")

def log(*args, **kwargs):
    """
    Logs data to the current Weights & Biases (wandb) training session.

    This function retrieves the current training session and calls its `log` method remotely.
    If no session is active or an error occurs during logging, an error message is logged.

    :param args: Positional arguments to pass to `wandb.log()`.
    :param kwargs: Keyword arguments to pass to `wandb.log()`.
    """
    if (training_session := get_current_session()) is not None:
        try:
            ray.get(training_session.log.remote(*args, **kwargs))
        except Exception as e:
            logger.error(f"Error logging to wandb: {e}")

def define_metric(*args, **kwargs):
    """
    Defines a metric for the current Weights & Biases (wandb) training session.

    This function retrieves the current training session and calls its `define_metric` method remotely.
    If no session is active or an error occurs, an error message is logged.
    It asserts that a training session must be active.

    :param args: Positional arguments to pass to `wandb.define_metric()`.
    :param kwargs: Keyword arguments to pass to `wandb.define_metric()`.
    """
    assert (training_session := get_current_session()) is not None
    try:
        ray.get(training_session.define_metric.remote(*args, **kwargs))
    except Exception as e:
        logger.error(f"Error defining metric to wandb: {e}")

def log_video(data: Dict[str, Any], video_key: str, fps: int):
    """
    Logs a video to the current Weights & Biases (wandb) training session.

    This function retrieves the current training session and calls its `log_video` method remotely.
    If no session is active or an error occurs during logging, an error message is logged.

    :param data: A dictionary containing the data to log. The video itself should be under the `video_key`.
    :param video_key: The key in the `data` dictionary that holds the video data.
    :param fps: The frames per second of the video.
    """
    if (training_session := get_current_session()) is not None:
        try:
            ray.get(training_session.log_video.remote(data, video_key, fps))
        except Exception as e:
            logger.error(f"Error logging video to wandb: {e}")