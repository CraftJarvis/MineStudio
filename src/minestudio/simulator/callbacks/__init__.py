'''
Date: 2024-11-11 07:53:19
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-05-09 14:27:14
FilePath: /MineStudio/minestudio/simulator/callbacks/__init__.py
'''
from minestudio.simulator.callbacks.callback import MinecraftCallback, Compose
from minestudio.simulator.callbacks.hard_reset import HardResetCallback # it must be the first callback
from minestudio.simulator.callbacks.speed_test import SpeedTestCallback
from minestudio.simulator.callbacks.record import RecordCallback
from minestudio.simulator.callbacks.summon_mobs import SummonMobsCallback
from minestudio.simulator.callbacks.mask_actions import MaskActionsCallback
from minestudio.simulator.callbacks.rewards import RewardsCallback
from minestudio.simulator.callbacks.fast_reset import FastResetCallback
from minestudio.simulator.callbacks.commands import CommandsCallback
from minestudio.simulator.callbacks.task import TaskCallback
from minestudio.simulator.callbacks.play import PlayCallback
from minestudio.simulator.callbacks.point import PointCallback, PlaySegmentCallback
from minestudio.simulator.callbacks.demonstration import DemonstrationCallback
from minestudio.simulator.callbacks.judgereset import JudgeResetCallback
from minestudio.simulator.callbacks.reward_gate import GateRewardsCallback
from minestudio.simulator.callbacks.voxels import VoxelsCallback
from minestudio.simulator.callbacks.init_inventory import InitInventoryCallback
from minestudio.simulator.callbacks.prev_action import PrevActionCallback

from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional, Literal

def load_callbacks_from_config(config: Union[str, Dict]):
    """
    Loads simulator callbacks from a configuration.

    This function iterates through registered simulator callbacks, creates instances
    from the provided configuration, and returns a list of initialized callbacks.

    :param config: The configuration source, which can be a file path (str) or a dictionary.
    :return: A list of initialized MinecraftCallback instances.
    """
    from minestudio.utils.register import Registers
    callbacks = []
    for key in Registers.simulator_callback.keys():
        callback = Registers.simulator_callback[key].create_from_conf(config)
        if callback is not None:
            callbacks.append(callback)
    return callbacks