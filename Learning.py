import os
import numpy as np
import argparse
from utils import save_result, Configuration
from Registry.AlgRegistry import alg_dict
from Registry.EnvRegistry import environment_dict
from Registry.TaskRegistry import task_dict
from Job.JobBuilder import default_params
from Environments.rendering import ErrorRender


def learn(config: Configuration):
    params = dict()
    for k, v in config.items():
        if k in alg_dict[config.algorithm].related_parameters():
            params[k] = v

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)

    rmsve = np.zeros((task_dict[config.task].num_of_policies(), config.num_steps, config.num_of_runs))
    for run in range(config.num_of_runs):
        random_seed = (run + config.num_of_runs) if config.rerun else run
        np.random.seed(random_seed)
        env = environment_dict[config.environment]()
        task = task_dict[config.task](run_number=run, num_steps=config.num_steps)
        agent = alg_dict[config.algorithm](task, **params)

        rmsve_of_run = np.zeros((task.num_policies, task.num_steps))
        agent.state = env.reset()
        error_render = ErrorRender(task.num_policies, task.num_steps)
        for step in range(task.num_steps):
            rmsve_of_run[:, step], error = agent.compute_rmsve()
            if config.render:
                error_render.add_error(error)
            agent.action = agent.choose_behavior_action()
            agent.next_state, r, is_terminal, info = env.step(agent.action)
            agent.learn(agent.state, agent.next_state, r, is_terminal)
            if is_terminal:
                agent.state = env.reset()
                agent.reset()
                continue
            agent.state = agent.next_state
            if config.render:
                env.render(mode='screen', render_cls=error_render)
        print(np.mean(rmsve_of_run, axis=0))
        rmsve[:, :, run] = rmsve_of_run
    rmsve_of_runs = np.transpose(np.mean(rmsve, axis=0))  # Average over all policies.
    save_result(config.save_path, '_RMSVE_mean_over_runs', np.mean(rmsve_of_runs, axis=0), params, config.rerun)
    save_result(config.save_path, '_RMSVE_stderr_over_runs',
                np.std(rmsve_of_runs, axis=0, ddof=1) / np.sqrt(config.num_of_runs), params, config.rerun)
    final_errors_mean_over_steps = np.mean(rmsve_of_runs[:, config.num_steps - int(0.01 * config.num_steps) - 1:],
                                           axis=1)
    save_result(config.save_path, '_mean_stderr_final',
                np.array([np.mean(final_errors_mean_over_steps), np.std(final_errors_mean_over_steps, ddof=1) /
                          np.sqrt(config.num_of_runs)]), params, config.rerun)
    auc_mean_over_steps = np.mean(rmsve_of_runs, axis=1)
    save_result(config.save_path, '_mean_stderr_auc',
                np.array([np.mean(auc_mean_over_steps),
                          np.std(auc_mean_over_steps, ddof=1) / np.sqrt(config.num_of_runs)]), params, config.rerun)

def learnControl(config: Configuration):
    params = dict()
    for k, v in config.items():
        if k in alg_dict[config.algorithm].related_parameters():
            params[k] = v

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)

    #rmsve = np.zeros((task_dict[config.task].num_of_policies(), config.num_steps, config.num_of_runs))
    steps = np.zeros((config.num_steps, config.num_of_runs))
    for run in range(config.num_of_runs):
        random_seed = (run + config.num_of_runs) if config.rerun else run
        np.random.seed(random_seed)
        env = environment_dict[config.environment]()
        task = task_dict[config.task](run_number=run, num_steps=config.num_steps)
        agent = alg_dict[config.algorithm](task, **params)
        print()

        agent.state = env.reset()
        #rmsve_of_run = np.zeros((task.num_policies, task.num_steps))
        #error_render = ErrorRender(task.num_policies, task.num_steps)
        for episode in range(task.num_steps):
            #rmsve_of_run[:, step], error = agent.compute_rmsve()
            #if config.render:
            #    error_render.add_error(error)
            agent.action = agent.choose_behavior_action(agent.state)
            while True:
                agent.next_state, r, is_terminal, info = env.step(agent.action)
                agent.next_action = agent.choose_behavior_action(agent.next_state)
                agent.learn(agent.state, agent.next_state, agent.action, agent.next_action, r, is_terminal)
                if is_terminal:
                    steps[episode, run] = agent.time_step
                    agent.state = env.reset()
                    agent.reset()
                    break

                if agent.time_step >= task.STEP_LIMIT:
                    steps[episode, run] = agent.time_step
                    agent.state = env.reset()
                    agent.reset()
                    print('Step Limit Exceeded!')
                    break
                agent.state = agent.next_state
                agent.action = agent.next_action
            #if config.render:
                #env.render(mode='screen', render_cls=error_render)
        #print(np.mean(rmsve_of_run, axis=0))
        #rmsve[:, :, run] = rmsve_of_run
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

    #save_result(config.save_path, '_steps_mean_over_runs', np.mean(steps, axis=0), params, config.rerun)

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
    if args.environment in ['MountainCar', 'DynaMaze', 'DynaMazeSwitch']:
        learnControl(config=config)
    else:
        learn(config=config)
