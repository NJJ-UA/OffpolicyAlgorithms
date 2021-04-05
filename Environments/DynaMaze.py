
class DynaMaze:
    def __init__(self, **kwargs):
        self._state = None

        # maze width
        self.WORLD_WIDTH = 9

        # maze height
        self.WORLD_HEIGHT = 6

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.ACTIONS = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state
        self.START_STATE = (2, 0)

        # goal state
        self.GOAL_STATES = [(0, 8)]

        # all obstacles
        self.obstacles = [(1, 2), (2, 2), (3, 2), (0, 7), (1, 7), (2, 7), (4, 5)]

    def reset(self):
        self._state = self.START_STATE
        return self._state

    def step(self, action):
        is_terminal = False
        x, y = self._state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = self._state
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
            is_terminal = True
        else:
            reward = 0.0
        self._state = (x, y)
        return self._state, reward, is_terminal, {}

