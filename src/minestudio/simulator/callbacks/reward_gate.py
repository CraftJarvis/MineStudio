import numpy as np
from minestudio.simulator.callbacks.callback import MinecraftCallback

class GateRewardsCallback(MinecraftCallback):
    """
    A callback for calculating rewards based on the formation of a Nether portal.

    This callback rewards the agent for building a valid Nether portal structure
    using obsidian blocks.
    """
    def __init__(self):
        """
        Initializes the GateRewardsCallback.
        """
        super().__init__()
        self.prev_info = {}
        self.reward_memory = {}
        self.current_step = 0
    
    def reward_as_smlest_pos(self, obsidian_position, obsidian_positions):
        """
        Calculates the reward for a potential portal frame based on a starting obsidian block.

        It checks for both X-fixed and Z-fixed portal orientations.

        :param obsidian_position: The (x, y, z) coordinates of a starting obsidian block.
        :param obsidian_positions: A list of (x, y, z) coordinates of all obsidian blocks.
        :return: The calculated reward for the best portal frame found from this starting block.
        """
        x, y, z = obsidian_position
        positive_pos = [(x, y, z), (x, y, z+1), (x, y, z+2), (x, y, z+3), 
                        (x, y+1, z+3), (x, y+2, z+3), (x, y+3, z+3), (x, y+4, z+3),
                        (x, y+4, z+2), (x, y+4, z+1), (x, y+4, z), 
                        (x, y+3, z), (x, y+2, z), (x, y+1, z)]
        negtive_pos = [(x, y+1, z+1), (x, y+1, z+2), (x, y+2, z+2), (x, y+3, z+2), (x, y+3, z+1), (x, y+2, z+1)]
        frame_num = len(set(positive_pos)&set(obsidian_positions))
        extra_bonus = max(0, frame_num-12)
        fix_x_reward = frame_num+extra_bonus - len(set(negtive_pos)&set(obsidian_positions)) - 0.1*len(set(obsidian_positions))
        
        #fix z reward
        positive_pos = [(x, y, z), (x+1, y, z), (x+2, y, z), (x+3, y, z),
                    (x+3, y+1, z), (x+3, y+2, z), (x+3, y+3, z), (x+3, y+4, z),
                    (x+2, y+4, z), (x+1, y+4, z), (x, y+4, z),
                    (x, y+3, z), (x, y+2, z), (x, y+1, z)]
        negtive_pos = [(x+1, y+1, z), (x+2, y+1, z), (x+2, y+2, z), (x+2, y+3, z), (x+1, y+3, z), (x+1, y+2, z)]
        frame_num = len(set(positive_pos)&set(obsidian_positions))
        #extra_bonus = max(0, frame_num-8) + max(0, frame_num-10) + 2*max(0, frame_num-12) + 4*max(0, frame_num-14)
        extra_bonus = max(0, frame_num-12)
        fix_z_reward = frame_num+extra_bonus - len(set(negtive_pos)&set(obsidian_positions)) - 0.1*len(set(obsidian_positions))
        
        larger_reward = max(fix_x_reward, fix_z_reward)
        return larger_reward
    
    def gate_reward(self, info, obs = {}):
        """
        Calculates the gate reward based on the current voxel information.

        It iterates through all obsidian blocks and finds the maximum possible
        portal reward.

        :param info: The info dictionary containing voxel data.
        :param obs: The observation dictionary (optional).
        :return: The maximum gate reward.
        """
        if "voxels" not in info:
            return 0
        voxels = info["voxels"]
        obsidian_positions = []
        
        for voxel in voxels:
            if "obsidian" in voxel["type"]:
                obsidian_positions.append((voxel["x"], voxel["y"], voxel["z"]))
        max_reward = 0
        for obsidian_position in obsidian_positions:
            reward = self.reward_as_smlest_pos(obsidian_position, obsidian_positions)
            max_reward = max(max_reward, reward)
        return max_reward   

    def after_reset(self, sim, obs, info):
        """
        Resets the current step count and previous reward.

        :param sim: The Minecraft simulator.
        :param obs: The observation from the simulator.
        :param info: Additional information from the simulator.
        :return: The observation and info.
        """
        self.current_step = 0
        self.prev_reward = 0
        return obs, info
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """
        Calculates the gate reward for the current step.

        The reward is the difference between the current gate reward and the previous
        gate reward (delta reward).

        :param sim: The Minecraft simulator.
        :param obs: The observation from the simulator.
        :param reward: The original reward from the simulator.
        :param terminated: Whether the episode has terminated.
        :param truncated: Whether the episode has been truncated.
        :param info: Additional information from the simulator.
        :return: The modified observation, overridden reward, terminated, truncated, and info.
        """
        override_reward = 0.
        cur_reward = self.gate_reward(info, obs)
        override_reward = cur_reward - self.prev_reward
        self.prev_reward = cur_reward
        self.current_step += 1
        return obs, override_reward, terminated, truncated, info