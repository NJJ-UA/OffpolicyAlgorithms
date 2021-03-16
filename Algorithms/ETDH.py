from Algorithms.ETD import ETD


class ETDH(ETD):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.beta = self.task.GAMMA

    def compute_step_size(self):
        if self.gamma == 1:
            #TODO: add time step
            return self.alpha
        else:
            return self.alpha/(self.lmbda+(1-self.lmbda)/(1-self.gamma))