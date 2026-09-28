# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

"""
minerl.herobraine.hero -- The interface between Hero (Malmo) and the minerl.herobraine package.
"""

import logging

logger = logging.getLogger(__name__)

import minestudio.simulator.minerl.herobraine.hero.mc
import minestudio.simulator.minerl.herobraine.hero.spaces

from minestudio.simulator.minerl.herobraine.hero.mc import KEYMAP
