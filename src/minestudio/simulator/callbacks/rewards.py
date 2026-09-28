'''
Date: 2024-11-11 17:44:15
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-14 20:09:56
FilePath: /Minestudio/minestudio/simulator/callbacks/rewards.py
'''

import numpy as np
from minestudio.simulator.callbacks.callback import MinecraftCallback

class RewardsCallback(MinecraftCallback):
    """
    A callback for calculating rewards based on in-game events.

    This callback allows defining custom rewards for various events, such as
    killing entities.
    """
    
    def __init__(self, reward_cfg):
        """
        Initializes the RewardsCallback.

        :param reward_cfg: A list of reward configurations.
                           Each configuration is a dictionary with keys:
                           "event": The type of event (e.g., "kill_entity").
                           "identity": A unique identifier for the reward.
                           "objects": A list of objects related to the event.
                           "reward": The reward value.
                           "max_reward_times": The maximum number of times this reward can be given.
        """
        super().__init__()
        """
        Examples:
            reward_cfg = [{
                "event": "kill_entity", 
                "identity": "kill sheep or cow", 
                "objects": ["sheep", "cow"], 
                "reward": 1.0, 
                "max_reward_times": 5, 
            }]
        """
        self.reward_cfg = reward_cfg
        self.prev_info = {}
        self.reward_memory = {}
        self.current_step = 0
    
    def after_reset(self, sim, obs, info):
        """
        Resets the reward memory and current step count.

        :param sim: The Minecraft simulator.
        :param obs: The observation from the simulator.
        :param info: Additional information from the simulator.
        :return: The observation and info.
        """
        self.prev_info = info.copy()
        self.reward_memory = {}
        self.current_step = 0
        return obs, info
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """
        Calculates and overrides the reward based on the reward configuration.

        :param sim: The Minecraft simulator.
        :param obs: The observation from the simulator.
        :param reward: The original reward from the simulator.
        :param terminated: Whether the episode has terminated.
        :param truncated: Whether the episode has been truncated.
        :param info: Additional information from the simulator.
        :return: The modified observation, overridden reward, terminated, truncated, and info.
        """
        override_reward = 0.
        for reward_info in self.reward_cfg:
            event_type = reward_info['event']
            delta = 0
            for obj in reward_info['objects']:
                delta += self._get_obj_num(info, event_type, obj) - self._get_obj_num(self.prev_info, event_type, obj)
                if delta <= 0:
                    continue
                already_reward_times = self.reward_memory.get(reward_info['identity'], 0)
                if already_reward_times < reward_info['max_reward_times']:
                    override_reward += reward_info['reward']
                    self.reward_memory[reward_info['identity']] = already_reward_times + 1
                break
        self.prev_info = info.copy()

        self.current_step += 1
        return obs, override_reward, terminated, truncated, info

    def _get_obj_num(self, info, event_type, obj):
        """
        Gets the number of objects of a specific type for a given event.

        :param info: The info dictionary.
        :param event_type: The type of event.
        :param obj: The object to count.
        :return: The number of objects.
        """
        if event_type not in info:
            return 0.
        if obj not in info[event_type]:
            return 0.
        res = info[event_type][obj]
        return res.item() if isinstance(res, np.ndarray) else res
