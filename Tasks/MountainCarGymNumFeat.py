import gym
import torch


class MountainCarGymNumFeat:
    def __init__(self,  **kwargs):
        self.env = gym.make('MountainCar-v0').unwrapped
        self.env.reset()
        self.num_episodes = kwargs.get('num_episodes', 50)
        self.device = kwargs.get('device')
        self.GAMMA = 1
        self.STEP_LIMIT = 1000

        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.shape[0]

        self.states_range = self.env.observation_space.high - self.env.observation_space.low

    def reset(self):
        return self.process_state(self.env.reset())

    #def get_state(self):
    #    return self.current_screen - self.last_screen

    def step(self, action):
        self.env.render()
        state, reward, is_terminal, info = self.env.step(action.item())
        reward = torch.tensor([reward], device=self.device)
        return self.process_state(state), reward, is_terminal, info

    def process_state(self, state):
        return torch.from_numpy((2*(state - self.env.observation_space.low)/self.states_range)-1).unsqueeze(0).to(self.device)
