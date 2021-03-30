from Algorithms.EQ import EQ


class EQH(EQ):

    def compute_step_size(self):
        if self.gamma == 1:
            return self.alpha/(self.time_step + 1)
        else:
            return self.alpha/(self.lmbda+(1-self.lmbda)/(1-self.gamma))