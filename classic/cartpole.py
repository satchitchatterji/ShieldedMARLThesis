# base code for openai cartpole
# skeleton code that can be used by any RL algorithm

import gymnasium as gym

import matplotlib.pyplot as plt
import numpy as np

from sarsa_agent import SARSAAgent
from q_agent import QAgent
from discretize import *
from config import config
from print_rewards import *

def run_episode(env, agent, max_steps):
    observation = env.reset()[0]
    totalreward = 0
    agent.begin_episode()
    for _ in range(max_steps):
        # env.render()
        if agent.observation_type == 'discrete':
            observation = discretize_state(observation)
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        agent.update_reward(reward)
        totalreward += reward
        if terminated or truncated:
            break
    return totalreward

def run(env, agent, run_fn, num_episodes):
    if not config.ready():
        raise Exception('Config not ready!')
    else:
        pass
        # print('Config ready!')
        # config.print()
    bestparams = None
    bestreward = None
    history = []
    # deterministic_history = []
    for _ in range(num_episodes):
        params = agent.get_params()
        reward = run_fn(env, agent, env.spec.max_episode_steps)
        # deterministic_history.append(evaluate(config.env, agent, params, env.spec.max_episode_steps, False))
        history.append(reward)
        if bestreward is None or reward > bestreward:
            bestreward = reward
            bestparams = params
    return bestparams, bestreward, np.array(history)

def evaluate(env_name, agent, params, max_steps, render=True):
    agent.eval_mode = True
    if render:
        env_render = gym.make(env_name, render_mode="human")
    else:
        env_render = gym.make(env_name)
    agent.set_params(params)
    reward = run_episode(env_render, agent, max_steps)
    env_render.close()
    agent.eval_mode = False
    return reward

if __name__ == '__main__':
    env = gym.make(config.env)
    config.update_action_space(env.action_space.n)
    config.update_observation_space(len(env.reset()[0]))
    histories = []
    # determinisitcs = []
    n_runs = 5
    for i in range(n_runs):
        agent = QAgent(config.num_states, env.action_space.n)

        bestparams, bestreward, history = run(env, agent, run_episode, config.num_episodes)

        eval_rewards = []
        for _ in range(10):
            eval_rewards.append(evaluate(config.env, agent, bestparams, env.spec.max_episode_steps, render=False))
        
        print('Run {}: Best train reward: {}, mean eval reward: {}, std eval reward: {}'.format(i, bestreward, np.mean(eval_rewards), np.std(eval_rewards)))
        env.close()
    
        histories.append(history)
        # determinisitcs.append(determinisitc)

    
    plot_all_multi_with_mean(histories, ['Run {}'.format(i) for i in range(n_runs)])
    # plot_all_multi_with_mean(determinisitcs, ['Run {}'.format(i) for i in range(n_runs)])