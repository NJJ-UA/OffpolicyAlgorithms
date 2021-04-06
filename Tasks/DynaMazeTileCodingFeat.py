import numpy as np
import random

from Environments.DynaMaze import DynaMaze
from Tasks.BaseTask import BaseTask
from utils import ImmutableDict


class DynaMazeTileCodingFeat(BaseTask, DynaMaze):
    def __init__(self, **kwargs):
        BaseTask.__init__(self)
        DynaMaze.__init__(self)

        self.tilecode = Tile(4, [8, 8], [0, self.WORLD_HEIGHT - 1], [0, self.WORLD_WIDTH - 1], [1, 3], len(self.ACTIONS))
        #self.feature_rep = self.load_feature_rep()
        self.num_features = self.tilecode.total_size
        self.num_steps = kwargs.get('num_steps', 100)
        self.GAMMA = 0.95
        self.STEP_LIMIT = 5000
        #self.behavior_dist = self.load_behavior_dist()
        #self.state_values = self.load_state_values()
        #self.ABTD_si_zero = 1
        #self.ABTD_si_max = 4


        #self.stacked_feature_rep = self.stack_feature_rep()
        #self._active_policies_cache = {}

    def get_state_action_feature_rep(self, s, a):
        return self.tilecode.tiles(s, a)

class Tile:
    def __init__(self, num_tiling, tiling_size, x_range, y_range, dis_vec, num_actions):
        self.num_tiling = num_tiling
        self.tiling_x, self.tiling_y = tiling_size
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.tile_width_x = (self.x_max - self.x_min) / (self.tiling_x - 1)
        self.tile_width_y = (self.y_max - self.y_min) / (self.tiling_y - 1)
        self.dis_x, self.dis_y = dis_vec
        self.total_size = self.tiling_x * self.tiling_y * self.num_tiling * num_actions
        #self.total_size = self.num_tiling * (self.numtiling+self.dis_x)*(self.numtiling+self.dis_y) * num_actions

    def get_offset(self, i, dis_vec):
        return (i * dis_vec) % self.num_tiling


    def tiles(self, states, action):
        x,y = states
        tiles = np.zeros(self.total_size)
        for i in range(self.num_tiling):
            x_min = self.x_min - self.tile_width_x + self.tile_width_x * self.get_offset(i, self.dis_x) / (
                    self.num_tiling - 1)
            y_min = self.y_min - self.tile_width_y + self.tile_width_y * self.get_offset(i, self.dis_y) / (
                    self.num_tiling - 1)
            x_ind = int((x - x_min) // self.tile_width_x)
            y_ind = int((y - y_min) // self.tile_width_y)
            if x_ind >= self.tiling_x:
                x_ind = self.tiling_x - 1
            if y_ind >= self.tiling_y:
                y_ind = self.tiling_y - 1
            #print(i)
            #print(self.tiling_x)
            #print(self.tiling_y)
            #print(x_ind)
            tiles[self.tiling_x * self.tiling_y * self.num_tiling*(action) + i*self.tiling_x * self.tiling_y + y_ind * self.tiling_x + x_ind] = 1
            #tiles[i*self.tiling_x * self.tiling_y + x_ind] = 1
            #tiles[i*self.tiling_x * self.tiling_y + y_ind] = 1
        #tiles[self.get_action_ind(action)] = 1
        return tiles
