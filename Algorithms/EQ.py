from Algorithms.BaseTDControl import BaseTDControl
import numpy as np

class EQ(BaseTDControl):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.F = 0
        self.M = 0

    def learn_single_policy(self, s, s_p, a, a_p, r, is_terminal):
        self.F = self.get_interest(s, a) + self.gamma * self.F
        self.M = self.lmbda * self.get_interest(s, a) + (1 - self.lmbda) * self.F
        x, x_p = self.get_features(s, s_p, a, a_p, is_terminal)
        self.z = self.gamma * self.lmbda * self.z + self.M * x
        delta = self.get_delta(r, x, x_p)
        self.w += self.compute_step_size() * delta * self.z

    def get_interest(self, s, a):
        return 1

    def reset(self):
        super().reset()
        self.F = 0
        self.M = 0

    def get_delta(self, r, x, x_p):
        return r + self.gamma * self.compute_Q_star() - np.dot(self.w, x)

    def compute_Q_star(self):
        values = []
        for action in self.task.ACTIONS:
            values.append(self.get_value(self.next_state, action))
        return np.max(values)