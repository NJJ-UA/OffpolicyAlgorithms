from abc import abstractmethod


class BaseProblem:
    def __init__(self, run_number=0):
        self.run_number = run_number
        self.num_steps = 5000
        self.feature_rep = None
        self.num_features = None
        self.GAMMA = None
        self.behavior_dist = None
        self.state_values = None
        self.num_policies = None

    @abstractmethod
    def load_feature_rep(self):
        raise NotImplementedError

    @abstractmethod
    def get_state_feature_rep(self, s):
        raise NotImplementedError

    @abstractmethod
    def create_feature_rep(self):
        raise NotImplementedError

    @abstractmethod
    def select_target_action(self, s, policy_id=0):
        raise NotImplementedError

    @abstractmethod
    def select_behavior_action(self, s):
        raise NotImplementedError

    @abstractmethod
    def get_pi(self, s, a):
        raise NotImplementedError

    @abstractmethod
    def get_mu(self, s, a):
        raise NotImplementedError

    @abstractmethod
    def load_behavior_dist(self):
        return self.behavior_dist

    @abstractmethod
    def load_state_values(self):
        return self.state_values
