'''
Date: 2024-11-11 19:29:45
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-11-12 00:12:11
FilePath: /MineStudio/minestudio/simulator/callbacks/task.py
'''
import random
from minestudio.simulator.callbacks.callback import MinecraftCallback

class TaskCallback(MinecraftCallback):
    """
    A callback for managing and assigning tasks in the Minecraft environment.

    This callback randomly selects a task from a predefined list of tasks
    after each environment reset and adds it to the observation.
    """
    
    def __init__(self, task_cfg):
        """
        Initializes the TaskCallback.

        :param task_cfg: A list of task configurations.
                         Each configuration is a dictionary with keys:
                         'name': The name of the task (e.g., 'chop tree').
                         'text': A descriptive text for the task (e.g., 'chop the tree').
        """
        super().__init__()
        self.task_cfg = task_cfg
    
    def after_reset(self, sim, obs, info):
        """
        Selects a random task and adds it to the observation after a reset.

        :param sim: The Minecraft simulator.
        :param obs: The observation from the simulator.
        :param info: Additional information from the simulator.
        :return: The modified observation and info.
        """
        task = random.choice(self.task_cfg)
        print(f"Switching to task: {task['name']}.")
        obs["task"] = task
        return obs, info