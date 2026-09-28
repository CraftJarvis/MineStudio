# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

import collections

import gym

from minestudio.simulator.minerl.herobraine.env_spec import EnvSpec
from minestudio.simulator.minerl.herobraine.env_specs.treechop_specs import Treechop
from minestudio.simulator.minerl.herobraine.env_specs.equip_weapon_specs import EquipWeapon
from minestudio.simulator.minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minestudio.simulator.minerl.herobraine.env_specs.navigate_specs import Navigate
from minestudio.simulator.minerl.herobraine.env_specs.obtain_specs import ObtainDiamondShovelEnvSpec
from minestudio.simulator.minerl.herobraine.wrappers import Obfuscated, Vectorized
from minestudio.simulator.minerl.herobraine.env_specs import basalt_specs
import os

# Must load non-obfuscated envs first!
# Publish.py depends on this order for black-listing streams
MINERL_TREECHOP_V0 = Treechop()

MINERL_NAVIGATE_V0 = Navigate(dense=False, extreme=False)
MINERL_NAVIGATE_EXTREME_V0 = Navigate(dense=False, extreme=True)
MINERL_NAVIGATE_DENSE_V0 = Navigate(dense=True, extreme=False)
MINERL_NAVIGATE_DENSE_EXTREME_V0 = Navigate(dense=True, extreme=True)

MINERL_OBTAIN_DIAMOND_SHOVEL_V0 = ObtainDiamondShovelEnvSpec()

MINERL_EQUIP_WEAPON_V0 = EquipWeapon()
MINERL_HUMAN_SURVIVAL_V0 = HumanSurvival()

MINERL_BASALT_FIND_CAVES_ENV_SPEC = basalt_specs.FindCaveEnvSpec()
MINERL_BASALT_MAKE_WATERFALL_ENV_SPEC = basalt_specs.MakeWaterfallEnvSpec()
MINERL_BASALT_PEN_ANIMALS_VILLAGE_ENV_SPEC = basalt_specs.PenAnimalsVillageEnvSpec()
MINERL_BASALT_VILLAGE_HOUSE_ENV_SPEC = basalt_specs.VillageMakeHouseEnvSpec()

# Register the envs.
ENVS = [env for env in locals().values() if isinstance(env, EnvSpec)]
for env in ENVS:
    try:
        if env.name not in gym.envs.registry.env_specs:
            env.register()
    except AttributeError:
        if env.name not in gym.envs.registry:
            env.register()
