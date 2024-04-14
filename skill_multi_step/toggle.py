import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from .goto_goal import GoTo_Goal

class Toggle(BaseSkill):
    def __init__(self, init_obs, target_obj):
        self.unpack_obs(self.obs)
        self.target_obj = target_obj
    
    def __call__(self):
        # if self.obs[self.target_pos[0], self.target_pos[1], 2] == 1: # open
        #     return None, True, False
        
        # get the coordinate of target_obj
        target_pos = tuple(np.argwhere(self.map==self.target_obj)[0])
        print(target_pos)
        action = GoTo_Goal(obs, target_pos)()
        return action, False, False