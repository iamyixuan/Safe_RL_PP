'''This script tests the RL implementation.'''

import shutil
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.envs.benchmark_env import Task, Environment

def task_generator(rng, low = 0, high = 1):
    scale = rng.uniform(low,high)
    #type = np.random.choice(['circle','square'])
    #type = 'circle'
    type = rng.choice(['random','circle'])
    #scale = scale * 2.0 if type=='random' else scale
    return scale,type

def run_episodes(ctrl, config, n_policy_updates, rng, low, high, env_step):
    '''Function to run PPO policy updates
    
    Args:
        ctrl: the controller
        config: configuration files
        n_policy_updates: Number of policy updates
        low : lower bound of random scale
        high : upper bound of random scale
    '''
    task = 'traj_tracking'
    env_func = partial(make,
                    config.task,
                    **config.task_config)
    ctrl.assign_steps(env_step)
    for m in range(n_policy_updates):
        scale, type = task_generator(rng, low, high)
        task_info = {
                    'trajectory_type': type,
                    'num_cycles': 2,
                    'trajectory_plane': 'zx',
                    'trajectory_position_offset': [0, 0],
                    'trajectory_scale': scale
            }
        env = env_func(task=task,task_info=task_info,randomized_init=False, init_state=np.zeros(4),random_traj_seed=rng.integers(low=1,high=1000))
        train_env = env_func(task=task,task_info=task_info,randomized_init=False, init_state=np.zeros(4),random_traj_seed=rng.integers(low=1,high=1000))
        # new 9/3
        ctrl.agent.train()
        experiment = BaseExperiment(env=env,ctrl=ctrl,train_env=train_env)
        experiment.launch_training()
        env.close()
        train_env.close()
        # new 9/4
        #ret.append(ctrl.ret)
        print(f"training step :{m+1}, total: {n_policy_updates}.")
        print(f"||||||||||||||||||||||||||||||||||||||DEVICE USED:{ctrl.device}, {next(ctrl.agent.ac.actor.parameters()).device}|||||||||||||||||||||||||||||||||||||||||||")
    #print(f"Training done for {n_policy_updates} episodes")

def violation_eval(hard_constraint,soft_constraint,soft_allowance,ctrl, config, lambda_sample_size, rng, low=0,high=1):
    ctrl.assign_steps(150)
    h_violation = 0
    s_violation = 0
    env_func = partial(make,
                config.task,
                **config.task_config)
    print(f"Starting evaluation of constraints")
    for j in range(lambda_sample_size):
        scale,type = task_generator(rng, low=low,high=high)
        task  = 'traj_tracking'
        task_info = {
            'trajectory_type': type,
            'num_cycles': 2,
            'trajectory_plane': 'zx',
            'trajectory_position_offset': [0, 0],
            'trajectory_scale': scale
        }   
        env = env_func(task=task,task_info=task_info,randomized_init=False, init_state=np.zeros(4),random_traj_seed=rng.integers(low=1,high=1000))
        # new 9/3
        ctrl.agent.eval()
        experiment = BaseExperiment(env=env, ctrl=ctrl)
        results, metrics = experiment.run_evaluation(n_episodes=1, n_steps=None, seeds=[j])
        h_violation += np.sum(np.abs(np.array(results['obs'][0][:, 0]))>hard_constraint)
        s_violation += np.sum(np.abs(np.diff(np.array(results['action'][0][:,0])))>soft_constraint)-soft_allowance>0
        env.close()
    h_violation = h_violation /lambda_sample_size
    s_violation = s_violation /lambda_sample_size
    print(f"Updating Lambdas: lamb_1_gradient={h_violation}, lamb_2_gradient={s_violation}")
    print(f"Policy update: {ctrl.policy_update_step}th step, Lambda update: {ctrl.lambda_training_counter}th step")
    print(f"current lambdas: {ctrl.lamb_1,ctrl.lamb_2}")
    return h_violation,s_violation



def run(gui=True, 
        n_episodes=1, 
        n_steps=None, 
        hard_constraint = 0.7, 
        soft_constraint = 0.05, 
        soft_allowance = 6, 
        hard_lr = 0.002,
        soft_lr = 0.002,
        seed = 42,
        lamb_1 = 0, 
        lamb_2=0, 
        lamb_update_steps=8, 
        policy_update = 5,
        delta=0.17, 
        lambda_sample_size = 20, 
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
    # New 9/3
    #ctrl.device = 'cuda'
    gamma_powered = ctrl.gamma** 150
    
    rng = np.random.default_rng(seed)

    lamb_1 = 0
    lamb_2 = 0
    lambs = np.zeros((policy_update*lamb_update_steps+1,2))
    scales = []
    # keep track of loss:
    value_loss = np.zeros(policy_update*(lamb_update_steps+1)+2)
    policy_loss = np.zeros(policy_update*(lamb_update_steps+1)+2)
    #ret = []
    # Remove temporary files and directories
    shutil.rmtree(f'{curr_path}/temp', ignore_errors=True)
    # pre-training
    # Here it does not really matter which task we do.
    
    
    for i in range(policy_update):
        ctrl.policy_update_step  += 1
        ctrl.lambda_training_counter = 0
        run_episodes(ctrl=ctrl,config=config,n_policy_updates=5,rng=rng,low=0.5,high=1,env_step=15000)
        
        #ctrl.load(f'{curr_path}/temp/model_best.pt')
        #stable = 0
        
        value_loss[(ctrl.policy_update_step-1)*(lamb_update_steps+1)+ctrl.lambda_training_counter]=ctrl.value_loss
        policy_loss[(ctrl.policy_update_step-1)*(lamb_update_steps+1)+ctrl.lambda_training_counter]=ctrl.policy_loss
        for k in range(lamb_update_steps):
            ctrl.assign_lambda(lamb_1,lamb_2)
            ctrl.lambda_training_counter += 1
            run_episodes(ctrl=ctrl, config=config, n_policy_updates=5, rng=rng, low=0.5, high=1, env_step=6000)
            value_loss[(ctrl.policy_update_step-1)*(lamb_update_steps+1)+ctrl.lambda_training_counter]=ctrl.value_loss
            policy_loss[(ctrl.policy_update_step-1)*(lamb_update_steps+1)+ctrl.lambda_training_counter]=ctrl.policy_loss
            #ctrl.load(f'{curr_path}/temp/model_best.pt')
            lamb_gradient_1, lamb_gradient_2 = violation_eval(hard_constraint=hard_constraint,soft_constraint=soft_constraint,
                                                              soft_allowance=soft_allowance,ctrl=ctrl,config=config,
                                                              lambda_sample_size=lambda_sample_size, rng = rng, low=0.5,high=1)
            '''
            if lamb_gradient_1 ==0 and lamb_gradient_2 == 0:
                stable += 1
            else:
                stable = 0
            if stable == 4 and ctrl.lambda_training_counter > 10:
                break
            '''
            lamb_1 += lamb_gradient_1 * hard_lr
            lamb_2 = max(lamb_2+ (lamb_gradient_2-gamma_powered*delta)* soft_lr,0) 
            lambs[i * lamb_update_steps + k+1] = np.array([lamb_1,lamb_2])        
            
    # Last training    
    #ctrl.load(f'{curr_path}/temp/model_best.pt')
    # new 9/5
    shutil.rmtree(f'{curr_path}/temp/best_model.pt', ignore_errors=True)
    ctrl.assign_lambda(lamb_1,lamb_2)
    run_episodes(ctrl=ctrl, config=config, n_policy_updates=150, rng=rng, low=0.5, high=1, env_step=3000)
    value_loss[-1] = ctrl.value_loss
    policy_loss[-1] = ctrl.policy_loss
    # new 9/4
    #ret_urn = np.array(ret)
    # analysis for larger scale
    # new 9/3
    ctrl.agent.eval()
    scale = 1
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
    results, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps, seeds=[42])
    step_h_violation = np.abs(np.array(results['state'][0][:,0]))>hard_constraint
    step_s_violation = np.abs(np.diff(np.array(results['action'][0][:,0])))>soft_constraint
    n_hard_violation = np.sum(step_h_violation)
    n_soft_violation = np.sum(step_s_violation)
    hard_violation_idx=np.where(step_h_violation==True)[0]
    soft_violation_idx=np.where(step_s_violation==True)[0]
    
    print(f"----------------------------Large Circle Task-----------------------")
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
    results_s, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps, seeds=[42])
    step_h_violation = np.abs(np.array(results_s['state'][0][:,0]))>hard_constraint
    step_s_violation = np.abs(np.diff(np.array(results_s['action'][0][:,0])))>soft_constraint
    n_hard_violation = np.sum(step_h_violation)
    n_soft_violation = np.sum(step_s_violation)
    hard_violation_idx=np.where(step_h_violation==True)[0]
    soft_violation_idx=np.where(step_s_violation==True)[0]
    
    print(f"----------------------------Small Circle Task-----------------------")
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
    # analysis for random task
    scale = 0.75
    task  = 'traj_tracking'
    task_info = {
                    'trajectory_type': 'random',
                    'num_cycles': 2,
                    'trajectory_plane': 'zx',
                    'trajectory_position_offset': [0, 0],
                    'trajectory_scale': scale
                }
    env_r = env_func(task=task,task_info=task_info,randomized_init = False, init_state=np.zeros(4), random_traj_seed=rng.integers(low=1,high=1000))
    experiment = BaseExperiment(env=env_r, ctrl=ctrl)
    results_r, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps, seeds=[42])
    step_h_violation = np.abs(np.array(results_r['state'][0][:,0]))>hard_constraint
    step_s_violation = np.abs(np.diff(np.array(results_r['action'][0][:,0])))>soft_constraint
    n_hard_violation = np.sum(step_h_violation)
    n_soft_violation = np.sum(step_s_violation)
    hard_violation_idx=np.where(step_h_violation==True)[0]
    soft_violation_idx=np.where(step_s_violation==True)[0]
    
    print(f"----------------------------Random Task-----------------------")
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




    print(f"----------------------------Results and Parameters-------------------------")
    print(f'final value loss: {value_loss[-1]} | final policy loss: {policy_loss[-1]}')

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

        _, ax8 = plt.subplots()
        ax8.plot(np.linspace(0, 20, results_r['obs'][0].shape[0]), results_r['obs'][0][:, 0], 'r--', label='RL Trajectory')
        ax8.plot(np.linspace(0, 20, env_r.X_GOAL[:, 0].shape[0]), env_r.X_GOAL[:, 0], 'b', label='Reference')
        ax8.plot(np.linspace(0, 20, results_r['obs'][0].shape[0]), [hard_constraint]*(results_r['obs'][0].shape[0]), 'g--', label='Hard_constraint')
        ax8.plot(np.linspace(0, 20, results_r['obs'][0].shape[0]), [-hard_constraint]*(results_r['obs'][0].shape[0]), 'g--', label='Hard_constraint')
        ax8.set_xlabel(r'Time')
        ax8.set_ylabel(r'X')
        ax8.set_box_aspect(0.5)
        ax8.legend(loc='upper right')


        _, ax9 = plt.subplots()
        ax9.plot(np.linspace(0, 20, results_r['action'][0].shape[0]-1), np.diff(np.array(results_r['action'][0][:, 0])), 'r--', label='RL Trajectory Action Change')
        ax9.plot(np.linspace(0, 20, results_r['action'][0].shape[0]-1), [soft_constraint]*(results_r['action'][0].shape[0]-1), 'g--', label='Soft_constraint')
        ax9.plot(np.linspace(0, 20, results_r['action'][0].shape[0]-1), [-soft_constraint]*(results_r['action'][0].shape[0]-1), 'g--', label='Soft_constraint')
        ax9.set_xlabel(r'Time')
        ax9.set_ylabel(r'X')
        ax9.set_box_aspect(0.5)
        ax9.legend(loc='upper right')

        _, ax10 = plt.subplots()
        ax10.plot(np.linspace(0, 20, value_loss.shape[0]), value_loss, 'r--', label='Value loss')
        ax10.set_xlabel(r'Time')
        ax10.set_ylabel(r'X')
        ax10.set_box_aspect(0.5)
        ax10.legend(loc='upper right')

        _, ax11 = plt.subplots()
        ax11.plot(np.linspace(0, 20, policy_loss.shape[0]), policy_loss, 'b--', label='Policy loss')
        ax11.set_xlabel(r'Time')
        ax11.set_ylabel(r'X')
        ax11.set_box_aspect(0.5)
        ax11.legend(loc='upper right')

        # new 9/4
        #_, ax12 = plt.subplots()
        #ax12.plot(np.linspace(0, 20, ret_urn.shape[0]), ret_urn, 'b--', label='Return')
        #ax12.set_xlabel(r'Time')
        #ax12.set_ylabel(r'X')
        #ax12.set_box_aspect(0.5)
        #ax12.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    env.close()
    env_s.close()
    env_r.close()
    return env.X_GOAL, results, metrics, lamb_1,lamb_2


if __name__ == '__main__':
    run()
