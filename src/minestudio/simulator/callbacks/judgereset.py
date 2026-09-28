'''
Date: 2025-06-12 19:46:03
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-06-12 19:47:07
FilePath: /MineStudio/minestudio/simulator/callbacks/judgereset.py
'''
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.simulator.utils import MinecraftGUI, GUIConstants
from minestudio.simulator.utils.gui import PointDrawCall
from minestudio.utils.register import Registers
from rich import print

import time
from typing import Dict, Literal, Optional, Callable, Tuple
import cv2

@Registers.simulator_callback.register
class JudgeResetCallback(MinecraftCallback):
    """Resets the environment if a time limit is reached or episode terminates.

    This callback monitors the number of steps taken in an episode.
    If the episode terminates naturally or if the step count exceeds `time_limit`,
    it forces a reset for the next step.

    :param time_limit: The maximum number of steps per episode before forcing a reset.
                       Defaults to 600.
    :type time_limit: int, optional
    """

    def create_from_conf(source: Dict) -> Optional['JudgeResetCallback']:
        """Creates a JudgeResetCallback from a configuration.
        
        :param source: Configuration source.
        :type source: Dict
        :returns: JudgeResetCallback instance or None if no valid configuration is found.
        :rtype: Optional[JudgeResetCallback]
        """
        if 'time_limit' not in source:
            print("[red]Missing 'time_limit' for JudgeResetCallback, skipping.[/red]")
            return None
        return JudgeResetCallback(time_limit=source['time_limit'])

    def __init__(self, time_limit: int = 600):
        """Initializes the JudgeResetCallback.

        :param time_limit: Maximum steps per episode.
        """
        super().__init__()
        self.time_limit = time_limit
        self.time_step = 0

    def after_reset(self, sim, obs: Dict, info: Dict) -> Tuple[Dict, Dict]:
        """Resets the internal step counter after an environment reset.

        :param sim: The simulator instance.
        :param obs: The initial observation after reset.
        :param info: The initial info dictionary after reset.
        :returns: The passed `obs` and `info`.
        :rtype: Tuple[Dict, Dict]
        """
        self.time_step = 0
        print("Environment reset:", self.time_step)
        return obs, info

    def after_step(self, sim, obs: Dict, reward: float, terminated: bool, truncated: bool, info: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Checks for termination or time limit and flags for reset if needed.

        Increments the step counter. If `terminated` is true or `self.time_step`
        exceeds `self.time_limit`, it sets `terminated` to True to signal
        a reset and resets `self.time_step`.

        :param sim: The simulator instance.
        :param obs: The observation after the step.
        :param reward: The reward received.
        :param terminated: Whether the episode has terminated.
        :param truncated: Whether the episode has been truncated.
        :param info: The info dictionary.
        :returns: The (potentially modified) obs, reward, terminated, truncated, and info.
        :rtype: Tuple[Dict, float, bool, bool, Dict]
        """
        self.time_step += 1
        if terminated or self.time_step > self.time_limit-1:
            print(f"Time limit reached, resetting the environment:", self.time_step)
            self.time_step = 0
            terminated = True
        return obs, reward, terminated, truncated, info