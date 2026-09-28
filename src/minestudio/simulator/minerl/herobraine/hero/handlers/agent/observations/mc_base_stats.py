# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

from collections import defaultdict
import jinja2
from typing import List, Any
import typing

from minestudio.simulator.minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minestudio.simulator.minerl.herobraine.hero.mc as mc
from minestudio.simulator.minerl.herobraine.hero import spaces
import numpy as np

__all__ = ['ObserveFromFullStats']


class ObserveFromFullStats(TranslationHandlerGroup):
    """
    Includes the use_item statistics for every item in MC that can be used
    """

    def xml_template(self) -> str:
        return str("""<ObservationFromFullStats/>""")

    def to_string(self) -> str:
        return self.stat_key

    def __init__(self, stat_key):
        assert stat_key is not None

        self.stat_key = stat_key
        super(ObserveFromFullStats, self).__init__(
            handlers=[_FullStatsObservation(statKeys) for statKeys in mc.ALL_STAT_KEYS if stat_key in statKeys]
        )

    def from_hero(self, x: typing.Dict[str, Any]) -> typing.Dict[str, Any]:
        if self.stat_key in x:
            ret = defaultdict(lambda: np.zeros((), dtype=float), x[self.stat_key])
        else:
            ret = defaultdict(lambda: np.zeros((), dtype=float))
        return ret

class _FullStatsObservation(KeymapTranslationHandler):
    def to_hero(self, x) -> int:
        for key in self.hero_keys:
            x = x[key]
        return x

    def __init__(self, key_list: List[str], space=None, default_if_missing=None):
        if space is None:
            if 'achievement' == key_list[0]:
                space = spaces.Box(low=0, high=1, shape=(), dtype=int)
            else:
                space = spaces.Box(low=0, high=100000000, shape=(), dtype=int)
        if default_if_missing is None:
            default_if_missing = np.zeros((), dtype=float)

        super().__init__(hero_keys=key_list, univ_keys=key_list, space=space,
                         default_if_missing=default_if_missing)

    def xml_template(self) -> str:
        return str("""<ObservationFromFullStats/>""")
