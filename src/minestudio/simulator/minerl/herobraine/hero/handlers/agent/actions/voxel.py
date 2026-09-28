import numpy as np

from minestudio.simulator.minerl.herobraine.hero.handlers.agent.action import Action
import minestudio.simulator.minerl.herobraine.hero.spaces as spaces


class VoxelAction(Action):
    """
    Handler which lets agents get voxel information

    Example usage:

    .. code-block:: python

        VoxelAction()

    To get block info within (x0, y0, z0) - (x1, y1, z1), use this action dictionary:

    .. code-block:: json

        {"voxels": [x0, x1, y0, y1, z0, z1]"}

    """

    def to_string(self):
        return 'voxels'

    def xml_template(self) -> str:
        return str("")

    def __init__(self):
        self._command = 'voxels'
        super().__init__(self.command, spaces.Box(-50, 50, [6], dtype = np.int32))

    def from_universal(self, x):
        return []

    def from_hero(self, info):
        return info["voxels"]

    def to_hero(self, x):
        """
        Returns a command string for the multi command action.
        :param x:
        :return:
        """
        cmd = f"voxels {x[0]},{x[1]},{x[2]},{x[3]},{x[4]},{x[5]}"

        return cmd