'''
Date: 2024-11-11 17:37:06
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-05-26 21:20:51
FilePath: /MineStudio/minestudio/simulator/callbacks/mask_actions.py
'''

from minestudio.simulator.callbacks.callback import MinecraftCallback

class MaskActionsCallback(MinecraftCallback):
    """Overrides specific action components with fixed values before each step.

    This callback can be used to constrain the agent's actions by forcing
    certain parts of the action dictionary to predefined values.

    :param action_settings: Keyword arguments where keys are action names (str)
                              and values are the fixed values for those actions.
                              Example: `MaskActionsCallback(attack=0)`
    """
    
    def __init__(self, **action_settings):
        """Initializes the MaskActionsCallback.

        :param action_settings: Action components to override and their fixed values.
        """
        super().__init__()
        self.action_settings = action_settings
    
    def before_step(self, sim, action: dict) -> dict:
        """Modifies the action dictionary before the simulator takes a step.

        Iterates through the `action_settings` provided during initialization
        and updates the corresponding keys in the `action` dictionary with the
        fixed values.

        :param sim: The simulator instance.
        :param action: The action dictionary to be modified.
        :type action: dict
        :returns: The modified action dictionary.
        :rtype: dict
        """
        for act, val in self.action_settings.items():
            action[act] = val
        return action