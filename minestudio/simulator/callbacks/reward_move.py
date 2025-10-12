'''
Date: 2024-11-11 17:44:15
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-14 20:09:56
FilePath: /Minestudio/minestudio/simulator/callbacks/rewards.py
'''

import numpy as np
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.simulator import MinecraftSim
class RewardsMoveCallback(MinecraftCallback):
    
    def __init__(self, instruction):
        super().__init__()
        self.instruction = instruction
        self.rewards = 0.0
        self.current_step = 0

    def after_reset(self, sim, obs, info):
        self.rewards = 0.0
        self.current_step = 0
        return obs, info
    
    def before_step(self, sim, action):
        penalty = action["forward"] + action["back"] + action["left"] + action["right"]
        if "forward" in self.instruction:
            self.reward = 4*action["forward"] - penalty
        elif "back" in self.instruction:
            self.reward = 4*action["back"] - penalty
        elif "left" in self.instruction:
            self.reward = 4*action["left"] - penalty
        elif "right" in self.instruction:
            self.reward = 4*action["right"] - penalty
        else:
            assert False, "not supported instruction"
        
        return action

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        override_reward = 0.
        self.rewards += self.reward
        self.current_step += 1

        if self.current_step % 100 == 90:
            override_reward = self.rewards
            self.rewards = 0.0
        return obs, override_reward, terminated, truncated, info


if __name__ == '__main__':
    # test if the simulator works

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation', default=False)
    args = parser.parse_args()
    
    if args.yes:
        check_engine(skip_confirmation=True)
    
    from minestudio.simulator.callbacks import SpeedTestCallback
    sim = MinecraftSim(
        action_type="env", 
        callbacks=[RewardsMoveCallback("left")]
    )
    obs, info = sim.reset()
    for i in range(200):
        action = sim.action_space.sample()
        obs, reward, terminated, truncated, info = sim.step(action)
        print(f"step {i}, reward: {reward}")
    sim.close()