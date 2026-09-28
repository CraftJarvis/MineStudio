'''
Date: 2024-11-11 17:26:22
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-04-19 15:42:35
FilePath: /MineStudio/var/minestudio/simulator/callbacks/summon_mobs.py
'''

from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils.register import Registers

@Registers.simulator_callback.register
class SummonMobsCallback(MinecraftCallback):
    """
    A callback for summoning mobs in the Minecraft world.

    This callback allows specifying the types, numbers, and spawn ranges of mobs
    to be summoned after each reset.
    """

    def create_from_conf(source):
        """
        Creates a SummonMobsCallback instance from a configuration source.

        :param source: The configuration source (e.g., file path or dictionary).
        :return: A SummonMobsCallback instance or None if 'summon_mobs' is not in the config.
        """
        data = MinecraftCallback.load_data_from_conf(source)
        if 'summon_mobs' in data:
            return SummonMobsCallback(data['summon_mobs'])
        else:
            return None

    def __init__(self, mobs) -> None:
        """
        Initializes the SummonMobsCallback.

        :param mobs: A list of mob configurations.
                     Each configuration is a dictionary with keys:
                     'name' or 'mob_name': The name of the mob (e.g., 'cow').
                     'number': The number of mobs to summon.
                     'range_x': A list or tuple specifying the [min, max] x-coordinate range for spawning.
                     'range_z': A list or tuple specifying the [min, max] z-coordinate range for spawning.
        """
        self.mobs = mobs
        """
        Examples:
            mobs = [{
                'name': 'cow', 
                'number': 10,
                'range_x': [-5, 5],
                'range_z': [-5, 5],
            }]
        """

    def after_reset(self, sim, obs, info):
        """
        Summons the specified mobs after the environment is reset.

        :param sim: The Minecraft simulator.
        :param obs: The observation from the simulator.
        :param info: Additional information from the simulator.
        :return: The modified observation and info after summoning mobs.
        """
        chats = []
        for mob in self.mobs:
            for _ in range(mob['number']):
                name = mob.get('name', mob.get('mob_name'))
                x = sim.np_random.uniform(*mob['range_x'])
                z = sim.np_random.uniform(*mob['range_z'])
                chat = f'/execute as @p at @p run summon minecraft:{name} ~{x} ~3 ~{z} {{Age:0}}'
                chats.append(chat)
        # chat.append('/effect give @e[type=#minecraft:is_animal] minecraft:slow_falling 99999 1 true')
        for chat in chats:
            obs, reward, done, info = sim.env.execute_cmd(chat)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info