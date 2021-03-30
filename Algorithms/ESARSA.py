from Algorithms.BaseTDControl import BaseTDControl


class ESARSA(BaseTDControl):
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

    def learn_multiple_policies(self, s, s_p, a, a_p, r, is_terminal):
        ...
        # delta, alpha_vec, *_, rho, stacked_x = super().learn_multiple_policies(s, s_p, r, is_terminal)
        # self.z = rho[:, None] * (self.lmbda * self.z * self.gamma_vec_t[:, None] + stacked_x)
        # self.w += (alpha_vec * delta)[:, None] * self.z
        # self.gamma_vec_t = self.gamma_vec_tp

    def get_interest(self, s, a):
        return 1

    def reset(self):
        super().reset()
        self.F = 0
        self.M = 0