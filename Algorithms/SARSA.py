from Algorithms.BaseTDControl import BaseTDControl


class SARSA(BaseTDControl):
    def learn_single_policy(self, s, s_p, a, a_p, r, is_terminal):
        delta, alpha, *_ = super().learn_single_policy(s, s_p, a, a_p, r, is_terminal)
        self.w += alpha * delta * self.z
