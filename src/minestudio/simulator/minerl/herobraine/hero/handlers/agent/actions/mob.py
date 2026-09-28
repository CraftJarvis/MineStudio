import numpy as np

from minestudio.simulator.minerl.herobraine.hero.handlers.agent.action import Action
import minestudio.simulator.minerl.herobraine.hero.spaces as spaces


class MobAction(Action):
    """
    Handler which lets agents get mob information

    Example usage:

    .. code-block:: python

        MobAction()

    To get mob info within (x0, y0, z0) - (x1, y1, z1), use this action dictionary:

    .. code-block:: json

        {"mobs": [x0, x1, y0, y1, z0, z1]"}

    """

    def to_string(self):
        return 'mobs'

    def xml_template(self) -> str:
        return str("")

    def __init__(self):
        self._command = 'mobs'
        super().__init__(self.command, spaces.Box(-1000, 1000, [6], dtype = np.int32))

    def from_universal(self, x):
        return []

    def from_hero(self, info):
        return info["mobs"]

    def to_hero(self, x):
        """
        Returns a command string for the multi command action.
        :param x:
        :return:
        """
        cmd = f"mobs {x[0]},{x[1]},{x[2]},{x[3]},{x[4]},{x[5]}"

        return cmd