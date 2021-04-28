import os
import numpy as np
import argparse
from utils import save_result, Configuration
from Registry.AlgRegistry import alg_dict
from Registry.EnvRegistry import environment_dict
from Registry.TaskRegistry import task_dict
from Job.JobBuilder import default_params
from Environments.rendering import ErrorRender

import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

def learn(config: Configuration):
    params = dict()
    for k, v in config.items():
        if k in alg_dict[config.algorithm].related_parameters():
            params[k] = v

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steps = np.zeros((config.num_steps, config.num_of_runs))
    for run in range(config.num_of_runs):
        random_seed = (run + config.num_of_runs) if config.rerun else run
        np.random.seed(random_seed)
        task = task_dict[config.task](run_number=run, num_episodes=config.num_steps, device=device)
        agent = alg_dict[config.algorithm](task, **params)

        # Initialize the environment and state
        agent.state = task.reset()
        for episode in range(task.num_episodes):
            while True:
                # Select and perform an action
                agent.action = agent.select_action(agent.state)

                agent.next_state, r, is_terminal, info = task.step(agent.action)

                agent.learn(r, is_terminal)

                if is_terminal:
                    steps[episode, run] = agent.time_step
                    agent.state = task.reset()
                    agent.reset()
                    break

                if agent.time_step >= task.STEP_LIMIT:
                    steps[episode, run] = agent.time_step
                    agent.state = task.reset()
                    agent.reset()
                    print('Step Limit Exceeded!')
                    break

                # Move to the next state
                agent.state = agent.next_state
        task.env.close()

    steps_of_runs = np.transpose(steps)
    save_result(config.save_path, '_steps_mean_over_runs', np.mean(steps_of_runs, axis=0), params, config.rerun)
    save_result(config.save_path, '_steps_stderr_over_runs',
                np.std(steps_of_runs, axis=0, ddof=1) / np.sqrt(config.num_of_runs), params, config.rerun)
    final_steps_mean_over_episodes = np.mean(steps_of_runs[:, config.num_steps - int(0.01 * config.num_steps) - 1:],
                                             axis=1)
    save_result(config.save_path, '_mean_stderr_final',
                np.array([np.mean(final_steps_mean_over_episodes), np.std(final_steps_mean_over_episodes, ddof=1) /
                          np.sqrt(config.num_of_runs)]), params, config.rerun)
    auc_mean_over_steps = np.mean(steps_of_runs, axis=1)
    save_result(config.save_path, '_mean_stderr_auc',
                np.array([np.mean(auc_mean_over_steps),
                          np.std(auc_mean_over_steps, ddof=1) / np.sqrt(config.num_of_runs)]), params, config.rerun)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', '-a', type=float, default=default_params['meta_parameters']['alpha'])
    parser.add_argument('--lmbda', '-l', type=float, default=default_params['meta_parameters']['lmbda'])
    parser.add_argument('--eta', '-et', type=float, default=default_params['meta_parameters']['eta'])
    parser.add_argument('--beta', '-b', type=float, default=default_params['meta_parameters']['beta'])
    parser.add_argument('--zeta', '-z', type=float, default=default_params['meta_parameters']['zeta'])
    parser.add_argument('--epsilon', '-eps', type=float, default=default_params['meta_parameters']['epsilon'])
    parser.add_argument('--tdrc_beta', '-tb', type=float, default=default_params['meta_parameters']['tdrc_beta'])
    parser.add_argument('--algorithm', '-alg', type=str, default=default_params['agent'])
    parser.add_argument('--task', '-t', type=str, default=default_params['task'])
    parser.add_argument('--num_of_runs', '-nr', type=int, default=default_params['num_of_runs'])
    parser.add_argument('--num_steps', '-ns', type=int, default=default_params['num_steps'])
    parser.add_argument('--sub_sample', '-ss', type=int, default=default_params['sub_sample'])
    parser.add_argument('--environment', '-e', type=str, default=default_params['environment'])
    parser.add_argument('--save_path', '-sp', type=str, default='-')
    parser.add_argument('--rerun', '-rrn', type=bool, default=False)
    parser.add_argument('--render', '-rndr', type=bool, default=False)
    args = parser.parse_args()
    if args.save_path == '-':
        args.save_path = os.path.join(os.getcwd(), 'Results', default_params['exp'], args.algorithm)

    config = Configuration(vars(args))
    print(args.environment)
    learn(config=config)
