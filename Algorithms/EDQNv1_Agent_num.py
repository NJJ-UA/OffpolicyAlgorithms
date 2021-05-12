import numpy as np
import random
import math
from collections import namedtuple
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class EDQNv1_Agent_num:
    def __init__(self, task, **kwargs):
        self.task = task
        self.gamma = self.task.GAMMA
        self.alpha = kwargs['alpha']
        self.BATCH_SIZE = 32
        self.EPS_START = 0.1
        self.EPS_END = 0.1
        self.EPS_DECAY = 5000
        self.TARGET_UPDATE = 100
        self.state, self.next_state, self.action, self.next_action = None, None, None, None
        self.time_step = 0
        self.steps_done = 0
        self.M = 0

        self.policy_net = DQN(self.task.n_states, self.task.n_actions).to(self.task.device)
        self.target_net = DQN(self.task.n_states, self.task.n_actions).to(self.task.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        #self.optimizer = optim.RMSprop(self.policy_net.parameters())
        #self.optimizer.lr = 0.011
        self.memory = ReplayMemory(2000)

    @staticmethod
    def related_parameters():
        return ['alpha']

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #print(type(self.policy_net(state)))
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.task.n_actions)]], device=self.task.device, dtype=torch.long)

    def learn(self, r, is_terminal):
        if is_terminal:
            self.next_state = None
        self.M = self.gamma * self.M + self.interest(self.state)
        self.memory.push(self.state, self.action, self.next_state, r, torch.tensor([self.M], device=self.task.device, dtype=torch.float))
        self.optimize_model()
        self.time_step += 1
        if self.steps_done % self.TARGET_UPDATE == 0:
            self.update_target()

    def interest(self, s):
        return 1

    def reset(self):
        self.M = 0
        self.time_step = 0

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.task.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        emphasis_batch = torch.cat(batch.emphasis)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.task.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduce = False)
        loss = loss.view(self.BATCH_SIZE)
        loss *= emphasis_batch
        loss = torch.mean(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def __str__(self):
        return f'agent:{type(self).__name__}'


class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.hNo = 50
        self.hidden = nn.Linear(inputs, self.hNo)
        nn.init.xavier_uniform_(self.hidden.weight, gain=nn.init.calculate_gain('relu'))
        self.head = nn.Linear(self.hNo, outputs)
        nn.init.xavier_uniform_(self.head.weight)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.hidden(x.float()))
        return self.head(x.view(x.size(0), -1))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'emphasis'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)