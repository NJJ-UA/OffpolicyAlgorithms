from Algorithms.BaseTDControl import BaseTDControl
import numpy as np

class Q(BaseTDControl):
    def learn_single_policy(self, s, s_p, a, a_p, r, is_terminal):
        delta, alpha, *_ = super().learn_single_policy(s, s_p, a, a_p, r, is_terminal)
        self.w += alpha * delta * self.z

    def learn_multiple_policies(self, s, s_p, a, a_p, r, is_terminal):
        ...
        # delta, alpha_vec, *_, rho, stacked_x = super().learn_multiple_policies(s, s_p, r, is_terminal)
        # self.z = rho[:, None] * (self.lmbda * self.z * self.gamma_vec_t[:, None] + stacked_x)
        # self.w += (alpha_vec * delta)[:, None] * self.z
        # self.gamma_vec_t = self.gamma_vec_tp

    def get_delta(self, r, x, x_p):
        return r + self.gamma * self.compute_Q_star() - np.dot(self.w, x)

    def compute_Q_star(self):
        values = []
        for action in self.task.ACTIONS:
            values.append(self.get_value(self.next_state, action))
        return np.max(values)
