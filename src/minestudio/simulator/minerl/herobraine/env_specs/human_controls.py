import abc
from abc import ABC
from minestudio.simulator.minerl.herobraine.hero.handlers.translation import TranslationHandler
from minestudio.simulator.minerl.herobraine.hero.handler import Handler

from minestudio.simulator.minerl.herobraine.hero import handlers as H, mc
from minestudio.simulator.minerl.herobraine.hero.mc import ALL_ITEMS, INVERSE_KEYMAP, SIMPLE_KEYBOARD_ACTION
from minestudio.simulator.minerl.herobraine.env_spec import EnvSpec

from typing import List
import numpy as np



class HumanControlEnvSpec(EnvSpec, ABC):
    """
    A simple base environment from which all other simple envs inherit.
    :param resolution:         resolution as (width, height) tuple at which minecraft
                               process generates POV (point of view) observations
    :param guiscale_range:     2 element tuple or list specifying range from which gui scale
                               is sampled. gui scale determines size of elements in minecraft
                               in-game gui like crafting. Note that gui scale is independent
                               of resolution, so gui elements at a fixed gui scale will appear
                               twice smaller if the resolution is increased by a factor of 2.

    :param gamma_range:        2 element tuple or list specifying range from which gamma
                               (parameter controlling brightness of POV observation) is sampled.
                               Default minecraft gamma is 0.0 (moody), reasonable values are between
                               0.0 and 2.0

    :param fov_range:          2 element tuple or list specifying range from which FOV (field of view)
                               angle is sampled. Default in minecraft is 70.0, 130 corresponds
                               "Quake" view.

    :param cursor_size_range:  2 element tuple or list specifying range of cursor size (in pixels).
                               Cursor is not rendered at all if cursor size is 0. When cursor size
                               is below 16, cursor sprite is rendered cropped.
    """


    def __init__(self, name, *args,
                 resolution=(640, 360),
                 guiscale_range=[1, 1],
                 gamma_range=[2.0, 2.0],
                 fov_range=[70.0, 70.0],
                 cursor_size_range=[16, 16],
                 rng=None,
                 **kwargs):

        self._rng = np.random if rng is None else rng
        self.resolution = resolution
        self.guiscale_range = guiscale_range
        self.gamma_range = gamma_range
        self.fov_range = fov_range
        self.cursor_size_range = cursor_size_range
        super().__init__(name, *args, **kwargs)

    def create_observables(self) -> List[TranslationHandler]:
        return [
            H.POVObservation(self.resolution),
            # H.InventoryObservation(ALL_ITEMS),
            H.PlainInventoryObservation()
        ]


    def create_actionables(self) -> List[TranslationHandler]:
        """
        Simple envs have some basic keyboard control functionality, but
        not all.
        """
        return [
           H.KeybasedCommandAction(v, v) for v in mc.KEYMAP.values()
        ] + [H.CameraAction()]

    def create_monitors(self) -> List[TranslationHandler]:
        return [H.IsGuiOpen(), H.ObservationFromCurrentLocation()]

    def create_agent_start(self) -> List[Handler]:
        gui_handler = H.GuiScale(self._rng.uniform(*self.guiscale_range))
        gamma_handler = H.GammaSetting(self._rng.uniform(*self.gamma_range))
        fov_handler = H.FOVSetting(self._rng.uniform(*self.fov_range))
        cursor_size_handler = H.FakeCursorSize(self._rng.randint(self.cursor_size_range[0], self.cursor_size_range[1] + 1))
        return [H.LowLevelInputsAgentStart(), gui_handler, gamma_handler, fov_handler, cursor_size_handler]


class SimpleHumanEmbodimentEnvSpec(HumanControlEnvSpec):
    """
    A simpler base environment for legacy support of MineRL tasks.
    """

    def __init__(self, name, *args, resolution=(64, 64), **kwargs):
        self.resolution = resolution
        kwargs["resolution"] = resolution
        super().__init__(name, *args, **kwargs)

    def create_observables(self) -> List[TranslationHandler]:
        return [
            H.POVObservation(self.resolution)
        ]

    def create_actionables(self) -> List[TranslationHandler]:
        """
        Simple envs have some basic keyboard control functionality, but
        not all.
        """
        return [
            H.KeybasedCommandAction(k, v) for k, v in INVERSE_KEYMAP.items()
            if k in SIMPLE_KEYBOARD_ACTION
        ] + [
            H.CameraAction(), H.CraftAction(
                ["none"] + mc.ALL_PERSONAL_CRAFTING_ITEMS,
                _other = "none", _default = "none"
            ), H.CraftNearbyAction(
                ["none"] + mc.ALL_CRAFTING_TABLE_ITEMS, 
                _other = "none", _default = "none"
            )
        ] + [
            H.EquipAction(
                ["none"] + mc.ALL_ITEMS, 
                _other = "none", _default = "none"
            ), H.PlaceBlock(
                ["none"] + mc.ALL_ITEMS, 
                _other = "none", _default = "none"
            )
        ] + [
            H.ChatAction()
        ] + [
            H.VoxelAction(), H.MobAction()
        ]

    def create_agent_start(self) -> List[Handler]:
        # These are needed to run the code
        gui_handler = H.GuiScale(self._rng.uniform(*self.guiscale_range))
        gamma_handler = H.GammaSetting(self._rng.uniform(*self.gamma_range))
        fov_handler = H.FOVSetting(self._rng.uniform(*self.fov_range))
        cursor_size_handler = H.FakeCursorSize(self._rng.randint(self.cursor_size_range[0], self.cursor_size_range[1] + 1))
        return [gui_handler, gamma_handler, fov_handler, cursor_size_handler]

    def create_monitors(self) -> List[TranslationHandler]:
        return []  # No monitors by default!
