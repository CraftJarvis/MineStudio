'''
Date: 2024-11-11 15:59:38
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-17 21:43:39
FilePath: /Minestudio/minestudio/simulator/callbacks/speed_test.py
'''
import time
from minestudio.simulator.callbacks.callback import MinecraftCallback

class SpeedTestCallback(MinecraftCallback):
    """
    A callback for testing the speed of the simulator.

    This callback measures the average time per step and the average FPS
    over a specified interval.
    """
    
    def __init__(self, interval: int = 100):
        """
        Initializes the SpeedTestCallback.

        :param interval: The interval (in steps) at which to print speed test status.
        """
        super().__init__()
        self.interval = interval
        self.num_steps = 0
        self.total_times = 0
    
    def before_step(self, sim, action):
        """
        Records the start time before executing a step.

        :param sim: The Minecraft simulator.
        :param action: The action to be executed.
        :return: The action.
        """
        self.start_time = time.time()
        return action
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """
        Calculates and prints the speed test status if the interval is reached.

        :param sim: The Minecraft simulator.
        :param obs: The observation from the simulator.
        :param reward: The reward from the simulator.
        :param terminated: Whether the episode has terminated.
        :param truncated: Whether the episode has been truncated.
        :param info: Additional information from the simulator.
        :return: The observation, reward, terminated, truncated, and info.
        """
        end_time = time.time()
        self.num_steps += 1
        self.total_times += end_time - self.start_time
        if self.num_steps % self.interval == 0:
            print(
                f'Speed Test Status: \n'
                f'Average Time: {self.total_times / self.num_steps :.2f} \n'
                f'Average FPS: {self.num_steps / self.total_times :.2f} \n'
                f'Total Steps: {self.num_steps} \n'
            )
        return obs, reward, terminated, truncated, info
