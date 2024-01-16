import matplotlib.pyplot as plt
import numpy as np

from prisoners_dilemma_ma import PrisonersDilemmaMAEnv as make_env

from discretize import *
from config import config
from print_rewards import *

from always_c_agent import CAgent
from always_d_agent import DAgent
from random_agent import RandomAgent
from tittat_agent import TitForTatAgent
from pd_q_agent import PDQAgent

def run_episode(env, agents, max_steps):
    observation = env.reset()[0]
    total_agent_rewards = np.zeros((len(agents), len(agents)))

    for agent in agents:
        agent.begin_episode()
    
    for _ in range(max_steps):
        # env.render()
        actions = []
        for i, agent in enumerate(agents):
            actions.append(agent.act(observation[i]))
        
        observation, rewards, terminated, truncated, info = env.step(actions)

        for i, agent in enumerate(agents):
            # Optimize individual reward
            agent.update_reward(rewards[i])

            # Optimize difference in rewards
            # equal_reward = (np.max(rewards) - np.min(rewards))**2
            # er_matrix = np.full((len(agents),len(agents)), equal_reward) - np.identity(len(agents))*equal_reward
            # agent.update_reward(er_matrix[i])

        # print(rewards, "\n")
        
        total_agent_rewards += rewards
        
        if terminated or truncated:
            break
    
    # return array of shape (num_agents, num_agents)
    # representing the total reward each agent received from each other agent
    return total_agent_rewards

def run(env, agents, run_fn, num_episodes):
    if not config.ready():
        raise Exception('Config not ready!')
    else:
        pass
        # print('Config ready!')
        # config.print()
    bestparams = [None]*num_episodes
    bestreward = [None]*num_episodes
    history = []
    for e in range(num_episodes):
        rewards = run_fn(env, agents, env.spec.max_episode_steps)
        history.append(rewards)
        params = [agent.get_params() for agent in agents]
        # find the best agent
        bestreward_idx = np.argmax(rewards.sum(axis=1))
        bestreward[e] = rewards[bestreward_idx].sum()
        bestparams[e] = params[bestreward_idx]

    return bestparams, bestreward, np.array(history)

def evaluate(env_name, agents, params, max_steps, render=True):
    for agent in agents:
        agent.eval_mode = True

    if render:
        env_render = make_env(agents, render_mode=True)
    else:
        env_render = make_env(agents, render_mode=False)

    # agent.set_params(params)
    reward = run_episode(env_render, agents, max_steps)
    env_render.close()

    for agent in agents:
        agent.eval_mode = True
        
    return reward

if __name__ == '__main__':
    histories = []
    epsilon_histories = []
    n_runs = config.n_runs
    best_q_params = [None, None]
    num_agents = 2
    
    for i in range(n_runs):
        
        agents = [
            PDQAgent(num_states=2, num_actions=2, num_agents=num_agents),
            # CAgent(num_states=2, num_actions=2, num_agents=num_agents),
            DAgent(num_states=2, num_actions=2, num_agents=num_agents),
            # PDQAgent(num_states=2, num_actions=2, num_agents=num_agents),
            # TitForTatAgent(num_states=2, num_actions=2, num_agents=num_agents),
        ]

        # if best_q_params[0] is not None:
        #     agents[0].set_params(best_q_params[0])
        #     agents[1].set_params(best_q_params[1])

        env = make_env(agents, render_mode=False)
        config.update_action_space(env.action_space.n)
        config.update_observation_space(len(env.reset()[0]))

        bestparams, bestreward, history = run(env, agents, run_episode, config.num_episodes)
        # print([bestparams.count()])
        # eval_rewards = []
        # for _ in range(10):
        #     eval_rewards.append(evaluate(config.env, agents, bestparams, env.spec.max_episode_steps, render=False))
        
        # print('Run {}: Best train reward: {}, mean eval reward: {}, std eval reward: {}'.format(i, bestreward, np.mean(eval_rewards), np.std(eval_rewards)))
        env.close()
        print('Run {}: Agent 0 params:'.format(i))
        print(agents[0].get_params())
        histories.append(history)
        epsilon_histories += agents[0].epsilon_histories

        # determinisitcs.append(determinisitc)

        # best_q_params = [agents[0].get_params(), agents[1].get_params()] # remove


    print(np.array(histories).shape) # (5, 1000, 2, 2) (n_runs, num_episodes, num_agents, num_agents)

    # concatenate runs
    histories = np.concatenate(histories, axis=0)
    print(histories.shape) #
    # plot concatenated histories in one plot and put a vertical line to sepatate runs
    sum_rewards_per_agent = np.sum(histories, axis=-1)/env.spec.max_episode_steps
    for agent in range(num_agents):
        plt.plot(sum_rewards_per_agent[:, agent], label='Agent {}'.format(agents[agent].name))
        plt.legend()
    # plot epsilon
    
    plt.vlines(np.arange(n_runs)*config.num_episodes, np.min(sum_rewards_per_agent), np.max(sum_rewards_per_agent), colors='k', linestyles='dashed')
    plt.show()
    plt.plot(epsilon_histories, label='epsilon', color='g')
    plt.show()
    # for run in range(n_runs):
    #     sum_rewards_per_agent = np.sum(histories[run], axis=-1)
    #     for agent in range(num_agents):
    #         plt.plot(sum_rewards_per_agent[:, agent], label='Agent {}'.format(agents[agent].name))
    #         plt.legend()
    #     plt.show()

    print([agent.name for agent in agents])