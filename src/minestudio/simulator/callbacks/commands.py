'''
Date: 2024-11-11 19:31:53
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-16 18:09:12
FilePath: /ROCKET-2/var/nfs-shared/shaofei/nfs-workspace/MineStudio/minestudio/simulator/callbacks/commands.py
'''
import os
import yaml
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional, Literal
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils.register import Registers

@Registers.simulator_callback.register
class CommandsCallback(MinecraftCallback):
    """
    Executes a list of Minecraft commands at specific lifecycle events.

    This callback is typically used to run setup commands after an environment reset.
    """
    
    def create_from_conf(source: Union[str, Dict]):
        """Creates a CommandsCallback instance from a configuration.

        The configuration can be a path to a YAML file or a dictionary.
        It should contain a list of command strings under the key 'custom_init_commands' or 'commands'.

        :param source: Configuration source (file path or dict).
        :type source: Union[str, Dict]
        :returns: A CommandsCallback instance or None if no commands are specified.
        :rtype: Optional[CommandsCallback]
        """
        data = MinecraftCallback.load_data_from_conf(source)
        available_keys = ['custom_init_commands', 'commands']
        for key in available_keys:
            if key in data:
                commands = data[key]
                return CommandsCallback(commands)
        return None
    
    def __init__(self, commands: List[str]):
        """Initializes the CommandsCallback with a list of commands.

        :param commands: A list of command strings to execute.
        :type commands: List[str]
        """
        super().__init__()
        self.commands = commands
    
    def after_reset(self, sim, obs: Dict, info: Dict) -> Tuple[Dict, Dict]:
        """Executes the configured commands after the environment resets.

        Each command is run sequentially, and the observation and info dictionaries
        are updated with the results from each command execution.

        :param sim: The simulator instance.
        :param obs: The current observation dictionary.
        :param info: The current info dictionary.
        :returns: Updated observation and info dictionaries.
        :rtype: Tuple[Dict, Dict]
        """
        for command in self.commands:
            _obs, reward, done, info = sim.env.execute_cmd(command)
            obs.update(_obs)
            info.update(info)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info

    def __repr__(self) -> str:
        """Returns a string representation of the CommandsCallback.

        :returns: String representation of the instance.
        :rtype: str
        """
        return f"CommandsCallback(commands={self.commands})"
