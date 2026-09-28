'''
Date: 2025-01-05 22:26:22
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-16 23:43:49
FilePath: /MineStudio/minestudio/simulator/callbacks/init_inventory.py
'''

import minecraft_data # https://github.com/SpockBotMC/python-minecraft-data  Provide easy access to minecraft-data in python
from typing import Union, List, Dict, Tuple, Set
import random
import json
import re
from pathlib import Path
from copy import deepcopy
from time import sleep
from rich import console
from minestudio.utils.register import Registers
from minestudio.simulator.callbacks.callback import MinecraftCallback

EQUIP_SLOTS = {
    "mainhand": 0,
    "offhand": 40,
    "head": 39,
    "chest": 38,
    "legs": 37,
    "feet": 36,
}
MIN_SLOT_IDX = 0
MAX_INVENTORY_IDX = 35
MAX_SLOT_IDX = 40
SLOT_IDX2NAME = {v: k for k, v in EQUIP_SLOTS.items()}
MIN_ITEMS_NUM = 0
MAX_ITEMS_NUM = 64

DISTRACTION_LEVEL = {"zero":[0],"one":[1],
                     "easy":range(3,7),"middle":range(7,16),"hard":range(16,35),
                     "normal":range(0,35)}



@Registers.simulator_callback.register
class InitInventoryCallback(MinecraftCallback):
    """Initializes the player's inventory at the start of an episode.

    This callback allows for precise control over the items the player starts with,
    including specifying items for exact slots, random slots, and adding
    distraction items. Quantities can be exact or defined by conditions.

    :param init_inventory: A list of dictionaries, each defining an item to initialize.
                           Each dictionary can specify "slot", "type", and "quantity".
                           Example: [{"slot": 0, "type": "oak_planks", "quantity": 64}]
    :type init_inventory: List[Dict]
    :param distraction_level: Defines the number of random distraction items.
                              Can be a string key (e.g., "easy", "hard") or a list of numbers.
                              Defaults to [0] (no distraction items).
    :type distraction_level: Union[List[int], str], optional
    """
    
    def create_from_conf(source: Union[str, Dict]):
        """Creates an InitInventoryCallback instance from a configuration source.

        Loads data from the given configuration (file path or dict) and
        initializes an InitInventoryCallback if 'init_inventory' is present.

        :param source: The configuration source.
        :type source: Union[str, Dict]
        :returns: An InitInventoryCallback instance or None.
        :rtype: Optional[InitInventoryCallback]
        """
        data = MinecraftCallback.load_data_from_conf(source)
        if 'init_inventory' in data:
            return InitInventoryCallback(data['init_inventory'])
        return None
    
    def __init__(self, init_inventory:List[Dict], distraction_level:Union[List[int],str]=[0]) -> None:
        """Initializes the InitInventoryCallback.

        :param init_inventory: Configuration for items to place in the inventory.
                               Example: `[{"slot": 0, "type": "oak_planks", "quantity": 64}]`
                               Quantity can support conditions like ">2", "<12,>10", "random".
        :type init_inventory: List[Dict]
        :param distraction_level: Controls the number of random items added as distractions.
                                  Can be a predefined level (str) or a list of possible counts.
        :type distraction_level: Union[List[int], str]
        """
        self.init_inventory = init_inventory
        self.distraction_level = DISTRACTION_LEVEL.get(distraction_level,[0]) if isinstance(distraction_level,str) else distraction_level
        
        mcd = minecraft_data("1.16")
        self.items_library = mcd.items_name
        self.items_names = list(mcd.items_name.keys())
        
    def after_reset(self, sim, obs: Dict, info: Dict) -> Tuple[Dict, Dict]:
        """Sets up the player's inventory after the environment resets.

        This method processes the `init_inventory` configuration:
        1.  Places items in specified slots.
        2.  Assigns items with "random" slots to available empty slots.
        3.  Adds distraction items based on `distraction_level`.
        4.  Uses Minecraft commands (`/replaceitem`) to modify the inventory.
        5.  Performs no-op actions and checks to ensure inventory setup is complete.

        :param sim: The simulator instance.
        :param obs: The current observation dictionary.
        :param info: The current info dictionary.
        :returns: The modified observation and info dictionaries.
        :rtype: Tuple[Dict, Dict]
        """
        chats = []
        visited_slots = set()
        uncertain_slots = [] 
        init_inventory = []
        print(self.init_inventory)
        for slot_info in self.init_inventory:
            slot = slot_info["slot"]
            if slot == "random":
                uncertain_slots.append(deepcopy(slot_info))
                continue
            visited_slots.add(int(slot))
            init_inventory.append(slot_info)
        unvisited_slots = set(range(MIN_SLOT_IDX, MAX_INVENTORY_IDX + 1)) - visited_slots
        
        # settle uncertain slots
        for uncertain_slot in uncertain_slots:
            slot = int(random.choice(list(unvisited_slots)))
            unvisited_slots.remove(slot)
            uncertain_slot["slot"] = slot
            init_inventory.append(uncertain_slot)
        
        # settle distraction slot
        distraction_num = min(random.choice(self.distraction_level),len(unvisited_slots))
        for _ in range(distraction_num):
            item_type = random.choice(self.items_names)
            slot = int(random.choice(list(unvisited_slots)))
            unvisited_slots.remove(slot)
            init_inventory.append({
                "slot":slot,
                "type":item_type,
                "quantity":"random",
            })
        self.slot_num = len(init_inventory)
        for item_dict in init_inventory:
            slot = item_dict["slot"]
            mc_slot =self._map_slot_number_to_cmd_slot(slot)
            item_type = item_dict["type"]
            assert item_type in self.items_names
            item_quantity = self._item_quantity_parser(item_dict["quantity"],int(self.items_library[item_type]["stackSize"]))
            chat = f"/replaceitem entity @p {mc_slot} minecraft:{item_type} {item_quantity}"
            if "metadata" in item_dict:
                chat += f" {item_dict['metadata']}"
            chats.append(chat)
        for chat in chats:
            obs, reward, done, info = sim.env.execute_cmd(chat)
        obs, info = sim._wrap_obs_info(obs, info)
        init_flag = False
        
        for _ in range(self.slot_num*2):
            action = sim.env.noop_action()
            obs, reward, done, info = sim.env.step(action)
            init_flag = self._check(obs)
            if init_flag:
                break
        if not init_flag:
            console.Console().log("[red]can't set up init inventory[/red]")
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info
    
    
    def _map_slot_number_to_cmd_slot(self,slot_number: Union[int,str]) -> str:
        """Maps an abstract slot number to a Minecraft command slot string.

        Converts a numerical slot (0-40) to its corresponding command selector
        (e.g., "weapon.mainhand", "armor.chest", "hotbar.0", "inventory.0").

        :param slot_number: The abstract slot number.
        :type slot_number: Union[int, str]
        :raises AssertionError: If `slot_number` is outside the valid range.
        :returns: The Minecraft command slot string.
        :rtype: str
        """
        slot_number = int(slot_number)
        assert MIN_SLOT_IDX <= slot_number <= MAX_SLOT_IDX, f"exceed slot index range:{slot_number}"
        if slot_number in {0, 40}:
            return f"weapon.{SLOT_IDX2NAME[slot_number]}"
        elif 36 <= slot_number <= 39:
            return f"armor.{SLOT_IDX2NAME[slot_number]}"
        elif 1 <= slot_number <= 8:
            return f"hotbar.{slot_number}"
        else:
            return f"inventory.{slot_number - 9}"

    def _item_quantity_parser(self,item_quantity: Union[int,str],max_items_num: int,one_p:float=0.7) -> int:
        """Parses item quantity from an integer or a string command.

        If `item_quantity` is an integer, it's returned directly.
        If it's a string, it can be:
        - "random": Returns 1 with probability `one_p`, else a random number up to `max_items_num`.
        - A comma-separated list of conditions (e.g., ">5,<=10"):
          Randomly chooses a quantity satisfying all conditions.

        :param item_quantity: The quantity to parse (int or string).
        :type item_quantity: Union[int, str]
        :param max_items_num: The maximum possible quantity for the item (stack size).
        :type max_items_num: int
        :param one_p: Probability of returning 1 when `item_quantity` is "random". Defaults to 0.7.
        :type one_p: float, optional
        :raises TypeError: If input is not an integer or string.
        :returns: The parsed item quantity.
        :rtype: int
        """
        
        if isinstance(item_quantity,str):
            
            candidate_nums: Set[int] = set(range(MIN_ITEMS_NUM, max_items_num + 1))
            
            if item_quantity == "random":
                one_flag = random.choices([True, False], weights=[one_p, 1 - one_p], k=1)[0]
                if one_flag:
                    return 1
                else:
                    return random.choice(list(candidate_nums))
            
            
            item_quantity_commands = item_quantity.split(",")
        
            def apply_command(op: str, val: int) -> Set[int]:
                """Helper to apply a single quantity condition."""
                return {
                    '<': set(range(MIN_ITEMS_NUM,val)),
                    '<=': set(range(MIN_ITEMS_NUM,val+1)),
                    '>': set(range(val+1,max_items_num+1)),
                    '>=': set(range(val,max_items_num+1)),
                    '==': {val}
                }[op]
        
            for item_quantity_command in item_quantity_commands:
                match = re.search(r'([<>]=?|==)\s*(\d+)', item_quantity_command.strip()) #matching "<...", ">...", "<=...", ">=...", "==..."
                if match:
                    operator, number = match.groups()
                    number = int(number)
                    candidate_nums &= apply_command(operator,number)
            if candidate_nums:
                item_quantity = random.choice(list(candidate_nums))
            else: # No valid quantity found, default to 1 or handle error appropriately
                item_quantity = 1 # Or raise an error
            
        elif not isinstance(item_quantity, int):
            raise TypeError("Input must be an integer or a string representing conditions")

        return item_quantity
    
    def _check(self,obs: Dict) -> bool:
        """Checks if the initial inventory setup was successful.

        Compares the number of non-empty slots in the current observation's
        inventory with the expected number of slots that should have been filled.

        :param obs: The current observation dictionary.
        :type obs: Dict
        :returns: True if the inventory appears to be set up correctly, False otherwise.
        :rtype: bool
        """
        current_slot_num = 0
        for slot_dict in obs["inventory"].values():
            if slot_dict["type"] != "none":
                current_slot_num+=1
        if current_slot_num >= self.slot_num:
            return True
        return False
    
if __name__ == "__main__":

    import numpy as np
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        SpeedTestCallback, 
        RecordCallback, 
        RewardsCallback, 
        TaskCallback,
        FastResetCallback
    )
    sim = MinecraftSim(
        action_type="env",
        callbacks=[
            SpeedTestCallback(50), 
            TaskCallback([
                {'name': 'craft', 'text': 'craft crafting_table'}, 
            ]),
            RecordCallback(record_path="./output", fps=30,record_actions=True,record_infos=True,record_origin_observation=True),
            RewardsCallback([{
                'event': 'craft_item', 
                'objects': ['crafting_table'], 
                'reward': 1.0, 
                'identity': 'craft crafting_table', 
                'max_reward_times': 1, 
            }]),
            FastResetCallback(
                biomes=['mountains'],
                random_tp_range=1000,
            ), 
            InitInventoryCallback([
                {"slot": 0,
                "type": "oak_planks",
                "quantity":1,},
                {"slot": 1,
                "type": "oak_planks",
                "quantity":">2",},
                {"slot": 2,
                "type": "oak_planks",
                "quantity":"<12,>10",},
                {"slot": "random",
                "type": "oak_planks",
                "quantity":"random",},
            ],distraction_level="normal")
        ]
    )
    obs, info = sim.reset()
    action = sim.noop_action()
    action["inventory"] = 1
    obs, reward, terminated, truncated, info = sim.step(action)
    for i in range(30):
        action = sim.noop_action()
        obs, reward, terminated, truncated, info = sim.step(action)
    sim.close()