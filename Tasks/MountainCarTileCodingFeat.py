import numpy as np
import random

from Environments.MountainCar import MountainCar
from Tasks.BaseTask import BaseTask
from utils import ImmutableDict


class MountainCarTileCodingFeat(BaseTask, MountainCar):
    def __init__(self, **kwargs):
        BaseTask.__init__(self)
        MountainCar.__init__(self)

        self.tilecode = Tile(5, [4, 4], [self.POSITION_MIN, self.POSITION_MAX], [self.VELOCITY_MIN, self.VELOCITY_MAX], [1, 3], 3)
        #self.feature_rep = self.load_feature_rep()
        self.num_features = self.tilecode.total_size
        self.num_steps = kwargs.get('num_steps', 100)
        self.GAMMA = 1.0
        self.STEP_LIMIT = 5000
        #self.behavior_dist = self.load_behavior_dist()
        #self.state_values = self.load_state_values()
        #self.ABTD_si_zero = 1
        #self.ABTD_si_max = 4


        self.num_policies = MountainCarTileCodingFeat.num_of_policies()
        #self.stacked_feature_rep = self.stack_feature_rep()
        #self._active_policies_cache = {}

    @staticmethod
    def num_of_policies():
        return 1

    # def get_terminal_policies(self, s):
    #     x, y = self.get_xy(s)
    #     terminal_policies = np.zeros(self.num_policies)
    #     for policy_id, condition in self.policy_terminal_condition.items():
    #         if condition(x, y):
    #             terminal_policies[policy_id] = 1
    #     return terminal_policies

    # def get_state_index(self, x, y):
    #     return int(y * np.sqrt(self.feature_rep.shape[0]) + x)
    #
    # def get_probability(self, policy_number, s, a):
    #     x, y = self.get_xy(s)
    #     probability = 0.0
    #     for condition, possible_actions in self.optimal_policies[policy_number]:
    #         if condition(x, y):
    #             if a in possible_actions:
    #                 probability = 1.0 / len(possible_actions)
    #                 break
    #     return probability
    #

    def get_state_action_feature_rep(self, s, a):
        return self.tilecode.tiles(s, a)

    # def get_active_policies(self, s):
    #     if s in self._active_policies_cache:
    #         return self._active_policies_cache[s]
    #     x, y = self.get_xy(s)
    #     active_policy_vec = np.zeros(self.num_policies)
    #     for policy_number, policy_values in self.optimal_policies.items():
    #         for (condition, _) in policy_values:
    #             if condition(x, y):
    #                 active_policy_vec[policy_number] = 1
    #                 break
    #     self._active_policies_cache[s] = active_policy_vec
    #     return active_policy_vec
    #
    # def load_feature_rep(self):
    #     return np.load(f'Resources/{self.__class__.__name__}/feature_rep.npy')
    #
    #
    # def create_feature_rep(self):
    #     ...
    #
    # def load_behavior_dist(self):
    #     return np.load(f'Resources/{self.__class__.__name__}/d_mu.npy')
    #
    # def load_state_values(self):
    #     return np.load(f'Resources/{self.__class__.__name__}/state_values.npy')
    #
    #
    # def get_mu(self, s, a):
    #     return np.ones(self.num_policies) * (1.0 / self.num_actions)
    #
    # def get_pi(self, s, a):
    #     pi_vec = np.zeros(self.num_policies)
    #     for policy_id, i in enumerate(self.get_active_policies(s)):
    #         if i:
    #             pi_vec[policy_id] = self.get_probability(policy_id, s, a)
    #     return pi_vec



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

    def get_action_ind(self, action):
        return self.total_size - 2 + action


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
            tiles[self.tiling_x * self.tiling_y * self.num_tiling*(action+1) + i*self.tiling_x * self.tiling_y + y_ind * self.tiling_x + x_ind] = 1
            #tiles[i*self.tiling_x * self.tiling_y + x_ind] = 1
            #tiles[i*self.tiling_x * self.tiling_y + y_ind] = 1
        #tiles[self.get_action_ind(action)] = 1
        return tiles
