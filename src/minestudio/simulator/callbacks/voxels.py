from minestudio.simulator.callbacks.callback import MinecraftCallback

class VoxelsCallback(MinecraftCallback):
    """
    A callback for specifying the voxel observation range.

    This callback adds a "voxels" instruction to the action, defining the
    volume from which voxel data should be collected.
    """
    
    def __init__(self, voxels_ins = [-7,7,-7,7,-7,7]):
        """
        Initializes the VoxelsCallback.

        :param voxels_ins: A list of 6 integers defining the voxel observation
                           range [xmin, xmax, ymin, ymax, zmin, zmax] relative
                           to the player. Defaults to [-7,7,-7,7,-7,7].
        """
        super().__init__()
        self.voxels_ins = voxels_ins

    def before_step(self, sim, action):
        """
        Adds the voxel instruction to the action before a step is taken.

        :param sim: The Minecraft simulator.
        :param action: The action dictionary.
        :return: The modified action dictionary with the "voxels" instruction.
        """
        action["voxels"] = self.voxels_ins
        return action