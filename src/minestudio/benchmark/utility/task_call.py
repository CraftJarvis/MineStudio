'''
Date: 2024-12-06 16:42:49
LastEditors: zhengxinyue
LastEditTime: 2024-12-11 17:44:10
FilePath: /MineStudio/minestudio/benchmark/utility/task.py
'''
import random
from minestudio.simulator.callbacks.callback import MinecraftCallback

class TaskCallback(MinecraftCallback):
    
    def __init__(self, task_cfg):
        """
        TaskCallback 
        Example:
            task_cfg = [{
                'name': 'chop tree',
                'text': 'chop the tree', 
            }]
        """
        super().__init__()
        self.task_cfg = task_cfg
    
    def after_reset(self, sim, obs, info):
        task = self.task_cfg
        print(f"Switching to task: {task['name']}.")
        obs["task"] = task
        return obs, info