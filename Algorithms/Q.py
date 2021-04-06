from Algorithms.BaseTDControl import BaseTDControl
import numpy as np

class Q(BaseTDControl):
    def learn_single_policy(self, s, s_p, a, a_p, r, is_terminal):
        delta, alpha, *_ = super().learn_single_policy(s, s_p, a, a_p, r, is_terminal)
        self.w += alpha * delta * self.z

    def get_delta(self, r, x, x_p):
        return r + self.gamma * self.compute_Q_star() - np.dot(self.w, x)

    def compute_Q_star(self):
        values = []
        for action in self.task.ACTIONS:
            values.append(self.get_value(self.next_state, action))
        return np.max(values)
