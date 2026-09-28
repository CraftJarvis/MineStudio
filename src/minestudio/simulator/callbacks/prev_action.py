'''
Date: 2024-11-11 19:31:53
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-06-12 19:48:23
FilePath: /MineStudio/minestudio/simulator/callbacks/prev_action.py
'''
import os
import yaml
import numpy as np
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional, Literal
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils.register import Registers

avail_keys = ["attack", "use", "inventory", "forward", "back", "left", "right",
        "sneak", "sprint", "jump", "drop", "hotbar.1", "hotbar.2", "hotbar.3",
        "hotbar.4", "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
        "camera"]

@Registers.simulator_callback.register
class PrevActionCallback(MinecraftCallback):
    """
    A callback that stores the previous action and adds it to the observation.

    This callback is useful for tasks where the agent needs to know its previous
    action to make a decision.
    """

    def create_from_conf(source):
        """Creates a PrevActionCallback from a configuration.

        Loads data from the source (file path or dict).

        :param source: Configuration source.
        :type source: Dict
        :returns: PrevActionCallback or None if no valid configuration is found.
        :rtype: Optional[PrevActionCallback]
        """
        if 'use_prev_action' in source and source['use_prev_action']:
            return PrevActionCallback()
        else:
            print("[red]use_prev_action is not set to True, skipping PrevActionCallback.[/red]")
            return None
    
    def __init__(self):
        """
        Initializes the PrevActionCallback.
        """
        super().__init__()
        self.prev_action = None

    def before_step(self, sim, action):
        """
        Stores the action before it is executed.

        :param sim: The Minecraft simulator.
        :param action: The action to be executed.
        :return: The action.
        """
        self.prev_action = action
        return action

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """
        Adds the previous action to the observation.

        :param sim: The Minecraft simulator.
        :param obs: The observation from the simulator.
        :param reward: The reward from the simulator.
        :param terminated: Whether the episode has terminated.
        :param truncated: Whether the episode has been truncated.
        :param info: Additional information from the simulator.
        :return: The modified observation, reward, terminated, truncated, and info.
        """
        obs["env_prev_action"] = {k: v for k, v in self.prev_action.items() if k in avail_keys}
        return obs, reward, terminated, truncated, info
    
    def after_reset(self, sim, obs, info):
        """
        Adds a default previous action to the observation after a reset.

        :param sim: The Minecraft simulator.
        :param obs: The observation from the simulator.
        :param info: Additional information from the simulator.
        :return: The modified observation and info.
        """
        obs["env_prev_action"] = {
            "attack": np.array(0), "use": np.array(0), "inventory": np.array(0), 
            "forward": np.array(0), "back": np.array(0), "left": np.array(0), "right": np.array(0), 
            "sneak": np.array(0), "sprint": np.array(0), "jump": np.array(0), "drop": np.array(0), 
            "hotbar.1": np.array(0), "hotbar.2": np.array(0), "hotbar.3": np.array(0), "hotbar.4": np.array(0), "hotbar.5": np.array(0), 
            "hotbar.6": np.array(0), "hotbar.7": np.array(0), "hotbar.8": np.array(0), "hotbar.9": np.array(0), 
            "camera": np.array([0, 0]), 
        }
        return obs, info
