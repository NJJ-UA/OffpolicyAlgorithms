import numpy as np
from Tasks.BaseTask import BaseTask

class BaseTDControl:
    def __init__(self, task: BaseTask, **kwargs):
        self.task = task
        self.w = np.zeros(self.task.num_features)
        self.z = np.zeros(self.task.num_features)
        self.gamma = self.task.GAMMA
        self.alpha = kwargs['alpha']
        self.lmbda = kwargs.get('lmbda')
        self.epsilon = kwargs.get('epsilon')
        #self.state_values = self.task.load_state_values()  # This is of size num_policies * 121
        #self.d_mu = self.task.load_behavior_dist()  # same size as state_values
        self.state, self.next_state, self.action, self.next_action = None, None, None, None
        self.time_step = 0

    @staticmethod
    def related_parameters():
        return ['alpha', 'lmbda', 'epsilon']

    # def compute_rmsve(self):
    #     est_value = np.dot(self.w, self.task.feature_rep.T)
    #     error = est_value - self.state_values
    #     error_squared = error * error
    #     return np.sqrt(np.sum(self.d_mu * error_squared.T, 0) / np.sum(self.d_mu, 0)), error

    def compute_step_size(self):
        return self.alpha

    def choose_behavior_action(self, s):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.choice(self.task.ACTIONS)
        values = []
        for action in self.task.ACTIONS:
            values.append(self.get_value(s, action))
        #return self.task.ACTIONS[np.argmax(values)]
        max_list = np.argwhere(values == np.amax(values))
        if len(max_list) == 0:
            return self.task.ACTIONS[np.argmax(values)]
        else:
            return self.task.ACTIONS[np.random.choice(max_list.flatten().tolist())]

    def choose_target_action(self, s):
        return self.choose_behavior_action(s)

    def learn(self, s, s_p, a, a_p, r, is_terminal):
        self.learn_single_policy(s, s_p, a, a_p, r, is_terminal)
        self.time_step += 1

    def get_features(self, s, s_p, a, a_p, is_terminal):
        x_p = np.zeros(self.task.num_features)
        if not is_terminal:
            x_p = self.task.get_state_action_feature_rep(s_p, a_p)
        x = self.task.get_state_action_feature_rep(s, a)
        return x, x_p

    def get_value(self, s, a):
        return np.dot(self.w, self.task.get_state_action_feature_rep(s, a))

    def get_isr(self, s):
        return 1
        pi = self.task.get_pi(s, self.action)
        mu = self.task.get_mu(s, self.action)
        rho = pi / mu
        return rho

    def get_delta(self, r, x, x_p):
        return r + self.gamma * np.dot(self.w, x_p) - np.dot(self.w, x)

    def learn_single_policy(self, s, s_p, a, a_p, r, is_terminal):
        x, x_p = self.get_features(s, s_p, a, a_p, is_terminal)
        rho = self.get_isr(s)
        alpha = self.compute_step_size()
        delta = self.get_delta(r, x, x_p)
        self.z = rho * (self.gamma * self.lmbda * self.z + x)
        return delta, alpha, x, x_p, rho

    def reset(self):
        self.z = np.zeros(self.task.num_features)
        self.time_step = 0

    def __str__(self):
        return f'agent:{type(self).__name__}'