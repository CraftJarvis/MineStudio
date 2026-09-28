'''
Date: 2024-11-11 16:15:32
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-16 23:45:32
FilePath: /MineStudio/minestudio/simulator/callbacks/hard_reset.py
'''
import random
import numpy as np
from typing import List
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils.register import Registers


@Registers.simulator_callback.register
class HardResetCallback(MinecraftCallback):
    """Performs a hard reset of the Minecraft environment.

    This callback forces a full environment reset by setting a specific seed
    and teleporting the player to a predefined spawn position. It is used
    when a complete and predictable reset is required.

    :param spawn_positions: A list of dictionaries, each specifying a "seed" (int)
                            and a "position" ([x, z, y] list) for spawning.
    :type spawn_positions: List[dict]
    """

    def create_from_conf(source):
        """Creates a HardResetCallback instance from a configuration source.

        Loads data from the given configuration (file path or dict) and
        initializes a HardResetCallback if 'spawn_positions' is present.

        :param source: The configuration source.
        :type source: any
        :returns: A HardResetCallback instance or None.
        :rtype: Optional[HardResetCallback]
        """
        data = MinecraftCallback.load_data_from_conf(source)
        if 'spawn_positions' in data:
            return HardResetCallback(data['spawn_positions'])
        else:
            return None

    def __init__(self, spawn_positions: List):
        """Initializes the HardResetCallback.

        :param spawn_positions: A list of potential spawn configurations.
                                Each configuration is a dict with "seed" and "position".
                                e.g., [{"seed": 123, "position": [0, 64, 0]}]
        :type spawn_positions: List[dict]
        """
        super().__init__()
        """
        position is a list of {
            "seed": int,
            "position": [x, z, y], 
        }
        """
        self.spawn_positions = spawn_positions

    def before_reset(self, sim, reset_flag):
        """Selects a spawn position and sets the environment seed before reset.

        Randomly chooses one of the provided `spawn_positions`, sets the
        environment's seed to the chosen seed, and forces a reset.

        :param sim: The simulator instance.
        :param reset_flag: The current reset flag status.
        :returns: True, to indicate that a reset should occur.
        :rtype: bool
        """
        self.position = random.choice(self.spawn_positions)
        sim.env.seed(self.position['seed'])
        return True

    def after_reset(self, sim, obs, info):
        """Teleports the player and allows the environment to settle after reset.

        After the environment resets, this method teleports the player to the
        selected x, z, y coordinates and then executes a number of no-op actions
        to allow the game world to stabilize.

        :param sim: The simulator instance.
        :param obs: The initial observation after reset.
        :param info: The initial info dictionary after reset.
        :returns: The modified observation and info.
        :rtype: tuple[dict, dict]
        """
        x, z, y = self.position["position"]
        obs, _, done, info = sim.env.execute_cmd(f"/tp @a {x} {z} {y}")
        for _ in range(50): 
            action = sim.env.action_space.no_op()
            obs, reward, done, info = sim.env.step(action)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info