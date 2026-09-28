'''
Date: 2024-11-11 16:15:32
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-06-12 19:47:39
FilePath: /MineStudio/minestudio/simulator/callbacks/fast_reset.py
'''
import random
import numpy as np
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils.register import Registers
from rich import print

@Registers.simulator_callback.register
class FastResetCallback(MinecraftCallback):
    """Implements a fast reset mechanism for the Minecraft simulator.

    This callback speeds up the reset process by executing a series of commands
    (e.g., /kill, /time set, /weather, /tp) instead of fully
    reinitializing the environment, if the environment has already been reset once.

    :param biomes: A list of biomes to randomly teleport to.
    :type biomes: list[str]
    :param random_tp_range: The range for random teleportation coordinates (x, z).
    :type random_tp_range: int
    :param start_time: The in-game time to set at reset, defaults to 0.
    :type start_time: int, optional
    :param start_weather: The weather to set at reset, defaults to 'clear'.
    :type start_weather: str, optional
    """

    def create_from_conf(source):
        """Creates a FastReset from a configuration.

        Loads data from the source (file path or dict).

        :param source: Configuration source.
        :type source: Dict
        :returns: FastResetCallback instance or None if no valid configuration is found.
        :rtype: Optional[FastResetCallback]
        """
        essential_keys = ['biomes', 'random_tp_range']
        for key in essential_keys:
            if key not in source:
                print(f"[red]Missing {key} for FastResetCallback, skipping.[/red]")
                return None
        return FastResetCallback(
            biomes=source['biomes'],
            random_tp_range=source['random_tp_range'],
            start_time=source.get('start_time', 0),
            start_weather=source.get('start_weather', 'clear')
        )

    def __init__(self, biomes, random_tp_range, start_time=0, start_weather='clear'):
        """Initializes the FastResetCallback.

        :param biomes: List of biomes for random teleportation.
        :param random_tp_range: Range for random teleportation coordinates.
        :param start_time: Initial in-game time after reset.
        :param start_weather: Initial weather after reset.
        """
        super().__init__()
        self.biomes = biomes
        self.random_tp_range = random_tp_range
        self.start_time = start_time
        self.start_weather = start_weather

    def before_reset(self, sim, reset_flag):
        """Performs a fast reset if the simulator has been reset before.

        If `sim.already_reset` is False (first reset), it allows the standard reset.
        Otherwise, it executes a sequence of commands to quickly reset the state:
        kills entities, sets time and weather, and teleports the player to a random
        biome and location.

        :param sim: The simulator instance.
        :param reset_flag: The current reset flag status.
        :returns: False if a fast reset was performed, otherwise `reset_flag`.
        :rtype: bool
        """
        if not sim.already_reset:
            return reset_flag
        biome = random.choice(self.biomes)
        x = np.random.randint(-self.random_tp_range // 2, self.random_tp_range // 2)
        z = np.random.randint(-self.random_tp_range // 2, self.random_tp_range // 2)
        fast_reset_commands = [
            "/kill", 
            f"/time set {self.start_time}",
            f"/weather {self.start_weather}",
            "/kill @e[type=!player]",
            "/kill @e[type=item]",
            f"/teleportbiome @a {biome} {x} ~0 {z}"
        ]
        for command in fast_reset_commands:
            obs, _, done, info = sim.env.execute_cmd(command)
        return False
