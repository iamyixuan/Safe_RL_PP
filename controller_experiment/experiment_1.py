'''This script tests the RL implementation.'''

import shutil
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.envs.benchmark_env import Task, Environment



def run(gui=True, 
        n_episodes=1, 
        n_steps=None, 
        hard_constraint = 0.6, 
        soft_constraint = 0.05, 
        soft_allowance = 6, 
        soft_lr = 3,
        lamb_1 = 0, 
        lamb_2=0, 
        lamb_update_steps=6, 
        policy_update = 9,
        delta=0.17, 
        lambda_sample_size = 1, 
        curr_path='.'):
    '''Main function to run RL experiments.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): How many steps to run the experiment.
        curr_path (str): The current relative path to the experiment folder.

    Returns:
        X_GOAL (np.ndarray): The goal (stabilization or reference trajectory) of the experiment.
        results (dict): The results of the experiment.
        metrics (dict): The metrics of the experiment.
    '''

    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()

    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    env_func = partial(make,
                       config.task,
                       **config.task_config)
    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
                output_dir=curr_path + '/temp')
    gamma_powered = ctrl.gamma** 150
    
    lamb_1 = 0
    lamb_2 = 0
    lambs = np.zeros((policy_update*lamb_update_steps+1,2))
    scales = []
    # Remove temporary files and directories
    shutil.rmtree(f'{curr_path}/temp', ignore_errors=True)
    for i in range(policy_update):
        # pre-training
        scale = np.random.uniform(0.4,1)
        task  = 'traj_tracking'
        task_info = {
                    'trajectory_type': 'circle',
                    'num_cycles': 2,
                    'trajectory_plane': 'zx',
                    'trajectory_position_offset': [0, 0],
                    'trajectory_scale': scale
            }
        scales.append(scale)
        ctrl.scale = scale
        ctrl.lambda_training = False
        ctrl.policy_update_step += 1
        env = env_func(task=task,task_info=task_info,randomized_init=False, init_state=np.zeros(4))
        train_env = env_func(task=task,task_info=task_info,randomized_init=False, init_state=np.zeros(4))
        ctrl.assign_steps(75000)
        experiment = BaseExperiment(env=env, ctrl=ctrl, train_env=train_env)
        experiment.launch_training()
        env.close()
        train_env.close()
        #ctrl.load(f'{curr_path}/temp/model_best.pt')
        stable = 0
        ctrl.lambda_training_counter = 1
        for k in range(lamb_update_steps):
            # Get initial state and create environments
            scale = np.random.uniform(0.4,1)
            task  = 'traj_tracking'
            task_info = {
                    'trajectory_type': 'circle',
                    'num_cycles': 2,
                    'trajectory_plane': 'zx',
                    'trajectory_position_offset': [0, 0],
                    'trajectory_scale': scale
            }
            scales.append(scale)
            ctrl.scale = scale
            ctrl.lambda_training = True
            env = env_func(task=task,task_info=task_info,randomized_init=False, init_state=np.zeros(4))
            train_env = env_func(task=task,task_info=task_info,randomized_init=False, init_state=np.zeros(4))

            # Create experiment, train, and run evaluation
            #ctrl.assign_steps(min(max(k//2,4),30)*1500)
            ctrl.assign_steps(9000)
            ctrl.assign_lambda(lamb_1,lamb_2)
            experiment = BaseExperiment(env=env, ctrl=ctrl, train_env=train_env)
            experiment.launch_training()
            ctrl.lambda_training_counter += 1
            #ctrl.load(f'{curr_path}/temp/model_best.pt')
            lamb_gradient_1 = 0
            lamb_gradient_2 = 0
            # Close environments
            env.close()
            train_env.close()
            # post analysis to update lambda
            for j in range(lambda_sample_size):
                scale = np.random.uniform(0.4,1)
                task  = 'traj_tracking'
                task_info = {
                    'trajectory_type': 'circle',
                    'num_cycles': 2,
                    'trajectory_plane': 'zx',
                    'trajectory_position_offset': [0, 0],
                    'trajectory_scale': scale
                }   
                env = env_func(task=task,task_info=task_info,randomized_init=False, init_state=np.zeros(4))
                experiment = BaseExperiment(env=env, ctrl=ctrl)
                results, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps, seeds=[j])
                lamb_gradient_1 += np.sum(np.abs(np.array(results['obs'][0][:, 0]))>hard_constraint)
                print(f"violated")
                print(np.sum(np.abs(np.diff(np.array(results['action'][0][:,0])))>soft_constraint)-soft_allowance>0)
                print(f"violation steps")
                print(np.where(np.abs(np.diff(np.array(results['action'][0][:,0])))>soft_constraint)[0])
                lamb_gradient_2 += np.sum(np.abs(np.diff(np.array(results['action'][0][:,0])))>soft_constraint)-soft_allowance>0
                print(lamb_gradient_2)
                env.close()
            lamb_gradient_1 = lamb_gradient_1/lambda_sample_size
            lamb_gradient_2 = max((lamb_gradient_2/lambda_sample_size) -  delta,0)
            if lamb_gradient_1 ==0 and lamb_gradient_2 == 0:
                stable += 1
            else:
                stable = 0
            if stable == 4 and ctrl.lambda_training_counter > 10:
                break
            lamb_1 += lamb_gradient_1
            lamb_2 = max(lamb_2 + soft_lr * lamb_gradient_2,0)
            lambs[i * lamb_update_steps + k+1] = np.array([lamb_1,lamb_2])        
            
    # Last training    
    #ctrl.load(f'{curr_path}/temp/model_best.pt')
    ctrl.assign_lambda(lamb_1,lamb_2)
    ctrl.assign_steps(105000)
    scale = 1
    ctrl.scale = scale
    ctrl.lambda_training = False
    task  = 'traj_tracking'
    task_info = {
                    'trajectory_type': 'circle',
                    'num_cycles': 2,
                    'trajectory_plane': 'zx',
                    'trajectory_position_offset': [0, 0],
                    'trajectory_scale': scale
                }
    env = env_func(task=task,task_info=task_info,randomized_init=False, init_state=np.zeros(4))
    train_env = env_func(task=task,task_info=task_info,randomized_init=False, init_state=np.zeros(4))
    experiment = BaseExperiment(env=env, ctrl=ctrl, train_env=train_env)
    experiment.launch_training()
    env.close()
    train_env.close()

    # analysis for larger scale
    n_hard_violation = np.zeros(lambda_sample_size)
    n_soft_violation = np.zeros(lambda_sample_size)
    hard_violation_idx = []
    soft_violation_idx = []
    for j in range(lambda_sample_size):
        scale = np.random.uniform(0.7,1)
        task  = 'traj_tracking'
        task_info = {
                        'trajectory_type': 'circle',
                        'num_cycles': 2,
                        'trajectory_plane': 'zx',
                        'trajectory_position_offset': [0, 0],
                        'trajectory_scale': scale
                    }
        env = env_func(task=task,task_info=task_info,randomized_init = False, init_state=np.zeros(4))
        experiment = BaseExperiment(env=env, ctrl=ctrl)
        results, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps, seeds=[j])
        step_h_violation = np.abs(np.array(results['state'][0][:,0]))>hard_constraint
        step_s_violation = np.abs(np.diff(np.array(results['action'][0][:,0])))>soft_constraint
        n_hard_violation[j] = np.sum(step_h_violation)
        n_soft_violation[j] = np.sum(step_s_violation)
        hard_violation_idx.append(np.where(step_h_violation==True)[0])
        soft_violation_idx.append(np.where(step_s_violation==True)[0])
        env.close()
    print(f"lambdas")
    print(lambs)
    print(f"number of hard violations")
    print(n_hard_violation)
    print(f"hard violation steps")
    print(hard_violation_idx)
    print(f"number of soft violations")
    print(n_soft_violation)
    print(f"soft violation steps")
    print(soft_violation_idx)
    print(f"scales used for training")
    print(scales)
    # analysis for smaller scale
    n_hard_violation_s = np.zeros(lambda_sample_size)
    n_soft_violation_s = np.zeros(lambda_sample_size)
    hard_violation_idx_s = []
    soft_violation_idx_s = []
    for j in range(lambda_sample_size):
        scale = 0.5
        task  = 'traj_tracking'
        task_info = {
                        'trajectory_type': 'circle',
                        'num_cycles': 2,
                        'trajectory_plane': 'zx',
                        'trajectory_position_offset': [0, 0],
                        'trajectory_scale': scale
                    }
        env_s = env_func(task=task,task_info=task_info,randomized_init = False, init_state=np.zeros(4))
        experiment = BaseExperiment(env=env_s, ctrl=ctrl)
        results_s, metrics_s = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps, seeds=[j])
        step_h_violation_s = np.abs(np.array(results_s['state'][0][:,0]))>hard_constraint
        step_s_violation_s = np.abs(np.diff(np.array(results_s['action'][0][:,0])))>soft_constraint
        n_hard_violation_s[j] = np.sum(step_h_violation_s)
        n_soft_violation_s[j] = np.sum(step_s_violation_s)
        hard_violation_idx_s.append(np.where(step_h_violation_s==True)[0])
        soft_violation_idx_s.append(np.where(step_s_violation_s==True)[0])
        env_s.close()
    ctrl.close()
    print(f"lambdas")
    print(lambs)
    print(f"number of hard violations")
    print(n_hard_violation_s)
    print(f"hard violation steps")
    print(hard_violation_idx_s)
    print(f"number of soft violations")
    print(n_soft_violation_s)
    print(f"soft violation steps")
    print(soft_violation_idx_s)
    print(f"scales used for training")
    print(scales)
    # Run experiment to learn the ultimate policy
    # ___________________________Coming Soon____________________________________
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    if gui is True:
        if system == Environment.CARTPOLE:
            graph1_1 = 2
            graph1_2 = 3
            graph3_1 = 0
            graph3_2 = 1
        elif system == 'quadrotor_2D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
        elif system == 'quadrotor_3D':
            graph1_1 = 6
            graph1_2 = 9
            graph3_1 = 0
            graph3_2 = 4

        _, ax = plt.subplots()
        ax.plot(results['obs'][0][:, graph1_1], results['obs'][0][:, graph1_2], 'r--', label='RL Trajectory')
        ax.scatter(results['obs'][0][0, graph1_1], results['obs'][0][0, graph1_2], color='g', marker='o', s=100, label='Initial State')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\dot{\theta}$')
        ax.set_box_aspect(0.5)
        ax.legend(loc='upper right')

        
        _, ax2 = plt.subplots()
        ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), results['obs'][0][:, 0], 'r--', label='RL Trajectory')
        ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), env.X_GOAL[:, 0], 'b', label='Reference')
        ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), [hard_constraint]*(results['obs'][0].shape[0]), 'g--', label='Hard_constraint')
        ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), [-hard_constraint]*(results['obs'][0].shape[0]), 'g--', label='Hard_constraint')
        ax2.set_xlabel(r'Time')
        ax2.set_ylabel(r'X')
        ax2.set_box_aspect(0.5)
        ax2.legend(loc='upper right')

        _, ax3 = plt.subplots()
        ax3.plot(results['obs'][0][:, graph3_1], results['obs'][0][:, graph3_2], 'r--', label='RL Trajectory')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(env.X_GOAL[:, graph3_1], env.X_GOAL[:, graph3_2], 'g--', label='Reference')
        ax3.scatter(results['obs'][0][0, graph3_1], results['obs'][0][0, graph3_2], color='g', marker='o', s=100, label='Initial State')
        ax3.set_xlabel(r'X')
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r'Vel')
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r'Z')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')
        


        _, ax4 = plt.subplots()
        ax4.plot(np.linspace(0, 20, results['action'][0].shape[0]-1), np.diff(np.array(results['action'][0][:, 0])), 'r--', label='RL Trajectory Action Change')
        ax4.plot(np.linspace(0, 20, results['action'][0].shape[0]-1), [soft_constraint]*(results['action'][0].shape[0]-1), 'g--', label='Soft_constraint')
        ax4.plot(np.linspace(0, 20, results['action'][0].shape[0]-1), [-soft_constraint]*(results['action'][0].shape[0]-1), 'g--', label='Soft_constraint')
        ax4.set_xlabel(r'Time')
        ax4.set_ylabel(r'X')
        ax4.set_box_aspect(0.5)
        ax4.legend(loc='upper right')


        _, ax5 = plt.subplots()
        ax5.plot(np.linspace(0, policy_update*lamb_update_steps, policy_update*lamb_update_steps+1), lambs[:,0], 'r--', label='Lambda 1')
        ax5.plot(np.linspace(0, policy_update*lamb_update_steps, policy_update*lamb_update_steps+1), lambs[:,1], 'b--', label='Lambda 2')
        ax5.set_xlabel(r'Time')
        ax5.set_ylabel(r'X')
        ax5.set_box_aspect(0.5)
        ax5.legend(loc='upper right')
        


        _, ax6 = plt.subplots()
        ax6.plot(np.linspace(0, 20, results_s['obs'][0].shape[0]), results_s['obs'][0][:, 0], 'r--', label='RL Trajectory')
        ax6.plot(np.linspace(0, 20, results_s['obs'][0].shape[0]), env_s.X_GOAL[:, 0], 'b', label='Reference')
        ax6.plot(np.linspace(0, 20, results_s['obs'][0].shape[0]), [hard_constraint]*(results_s['obs'][0].shape[0]), 'g--', label='Hard_constraint')
        ax6.plot(np.linspace(0, 20, results_s['obs'][0].shape[0]), [-hard_constraint]*(results_s['obs'][0].shape[0]), 'g--', label='Hard_constraint')
        ax6.set_xlabel(r'Time')
        ax6.set_ylabel(r'X')
        ax6.set_box_aspect(0.5)
        ax6.legend(loc='upper right')


        _, ax7 = plt.subplots()
        ax7.plot(np.linspace(0, 20, results_s['action'][0].shape[0]-1), np.diff(np.array(results_s['action'][0][:, 0])), 'r--', label='RL Trajectory Action Change')
        ax7.plot(np.linspace(0, 20, results_s['action'][0].shape[0]-1), [soft_constraint]*(results_s['action'][0].shape[0]-1), 'g--', label='Soft_constraint')
        ax7.plot(np.linspace(0, 20, results_s['action'][0].shape[0]-1), [-soft_constraint]*(results_s['action'][0].shape[0]-1), 'g--', label='Soft_constraint')
        ax7.set_xlabel(r'Time')
        ax7.set_ylabel(r'X')
        ax7.set_box_aspect(0.5)
        ax7.legend(loc='upper right')



        plt.tight_layout()
        plt.show()
    return env.X_GOAL, results, metrics, lamb_1,lamb_2


if __name__ == '__main__':
    run()
