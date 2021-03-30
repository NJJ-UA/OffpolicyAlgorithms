import numpy as np


class MountainCar:
    def __init__(self, **kwargs):
        #assert start_state_number < states_number, "start states numbers should be less than state number"

        #self._states_number = states_number
        #self._start_state_number = start_state_number
        #self._terminal = self._states_number
        self._state = None

        # all possible actions
        self.ACTION_REVERSE = -1
        self.ACTION_ZERO = 0
        self.ACTION_FORWARD = 1
        # order is important
        self.ACTIONS = [self.ACTION_REVERSE, self.ACTION_ZERO, self.ACTION_FORWARD]
        # bound for position and velocity
        self.POSITION_MIN = -1.2
        self.POSITION_MAX = 0.5
        self.VELOCITY_MIN = -0.07
        self.VELOCITY_MAX = 0.07


    def reset(self):
        self._state = (np.random.uniform(-0.6, -0.4), 0.0)
        return self._state

    def step(self, action):
        is_terminal = False
        new_velocity = self._state[1] + 0.001 * action - 0.0025 * np.cos(3 * self._state[0])
        new_velocity = min(max(self.VELOCITY_MIN, new_velocity), self.VELOCITY_MAX)
        new_position = self._state[0] + new_velocity
        new_position = min(max(self.POSITION_MIN, new_position), self.POSITION_MAX)
        if new_position == self.POSITION_MIN:
            new_velocity = 0.0
        if new_position == self.POSITION_MAX:
            is_terminal = True
        self._state = (new_position, new_velocity)
        return self._state, -1.0, is_terminal, {}