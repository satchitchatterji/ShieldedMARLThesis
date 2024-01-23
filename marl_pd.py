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
from pd_sarsa_agent import PDSARSAAgent

def run_episode(env, agents, max_steps):
    observation = env.reset()[0]
    total_agent_rewards = np.zeros((len(agents), len(agents)))

    for agent in agents:
        agent.begin_episode()
    
    for _ in range(max_steps):
        # env.render()
        actions = []
        for i, agent in enumerate(agents):
            actions.append(agent.act(np.array(observation).T[i]))
        
        observation, rewards, terminated, truncated, info = env.step(actions)

        for i, agent in enumerate(agents):
            # Optimize individual reward
            agent.update_reward(rewards[i])

            # Optimize social welfare
            # equal_reward = np.sum(rewards)
            # er_matrix = np.full((len(agents),len(agents)), equal_reward) - np.identity(len(agents))*equal_reward
            # agent.update_reward(er_matrix[i])

            # Optimize minimal difference in rewards
            # equal_reward = -(np.max(rewards) - np.min(rewards))**2
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


def run_synch(env, agents, run_fn, num_episodes):
    if not config.ready():
        raise Exception('Config not ready!')
    else:
        pass
        # print('Config ready!')
        # config.print()
    best_q_params = None
    bestparams = [None]*num_episodes
    bestreward = [None]*num_episodes
    history = []
    for e in range(num_episodes):
        
        for agent in agents:
            if agent.name == 'Q-Learning' or agent.name == 'SARSA':
                if best_q_params is not None:
                    agent.set_params(best_q_params)

        rewards = run_fn(env, agents, env.spec.max_episode_steps)
        history.append(rewards)
        params = [agent.get_params() for agent in agents]
        # find the best agent
        bestreward_idx = np.argmax(rewards.sum(axis=1))
        bestreward[e] = rewards[bestreward_idx].sum()
        bestparams[e] = params[bestreward_idx]

        agent_params = np.zeros(agents[0].get_params().shape)
        for agent in agents:
            if agent.name == 'Q-Learning' or agent.name == 'SARSA':
                agent_params += agent.get_params()
    
        best_q_params = agent_params/num_agents


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
    num_agents = 3

    for i in range(n_runs):
        
        agents = [
            PDQAgent(num_states=2, num_actions=2),
            PDQAgent(num_states=2, num_actions=2),
            PDSARSAAgent(num_states=2, num_actions=2),
            # CAgent(num_states=2, num_actions=2),
            # CAgent(num_states=2, num_actions=2),
            # CAgent(num_states=2, num_actions=2),
            # CAgent(num_states=2, num_actions=2),
            # CAgent(num_states=2, num_actions=2),
            # DAgent(num_states=2, num_actions=2),
            # TitForTatAgent(num_states=2, num_actions=2),
            # RandomAgent(num_states=2, num_actions=2)
        ]

        num_agents = len(agents)
        for agent in agents:
            agent.update_n_agents(num_agents)

        env = make_env(agents, render_mode=config.render_mode)
        config.update_action_space(env.action_space.n)
        config.update_observation_space(len(env.reset()[0]))

        bestparams, bestreward, history = run(env, agents, run_episode, config.num_episodes)
        # print([bestparams.count()])
        # eval_rewards = []
        # for _ in range(10):
        #     eval_rewards.append(evaluate(config.env, agents, bestparams, env.spec.max_episode_steps, render=False))
        
        # print('Run {}: Best train reward: {}, mean eval reward: {}, std eval reward: {}'.format(i, bestreward, np.mean(eval_rewards), np.std(eval_rewards)))
        env.close()
        histories.append(history)
        if agents[0].name == 'Q-Learning' or agents[0].name == 'SARSA':
            print('Run {}: Agent 0 params:'.format(i))
            print(agents[0].get_params())
            epsilon_histories += agents[0].epsilon_histories
        
        # determinisitcs.append(determinisitc)

    print(np.array(histories).shape) # (5, 1000, 2, 2) (n_runs, num_episodes, num_agents, num_agents)

    # concatenate runs
    histories = np.concatenate(histories, axis=0)
    print(histories.shape) #
    # # plot concatenated histories in one plot and put a vertical line to sepatate runs
    sum_rewards_per_agent = (np.sum(histories, axis=-1)/env.spec.max_episode_steps)/(num_agents-1)
    for agent in range(num_agents):
        plt.plot(sum_rewards_per_agent[:, agent], label='Agent {}'.format(agents[agent].name))
    plt.vlines(np.arange(n_runs)*config.num_episodes, np.min(sum_rewards_per_agent), np.max(sum_rewards_per_agent), colors='k', linestyles='dashed', label="Reset point")
    plt.title("Mean Utility per Agent per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Utility attained per opponent")
    plt.legend()
    plt.show()
    
    # # plot epsilon
    
    # if agents[0].name == 'Q-Learning':
    #     plt.plot(epsilon_histories, label='epsilon', color='g')
    #     plt.show()

    # plot sum of all agents' rewards per run
    sum_rewards_per_agent_all_games = (np.sum(histories, axis=-1)/env.spec.max_episode_steps)
    summed_histories = np.sum(sum_rewards_per_agent_all_games, axis=-1)
    plt.plot(summed_histories)
    plt.vlines(np.arange(n_runs)*config.num_episodes, np.min(summed_histories), np.max(summed_histories), colors='k', linestyles='dashed', label="Reset point")
    plt.legend()
    plt.title("Mean Social Welfare for All Games per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Utility")
    plt.show()

    # n choose 2 games
    total_games_per_episode = num_agents*(num_agents-1)/2
    summed_histories = summed_histories/total_games_per_episode
    plt.plot(summed_histories)
    plt.vlines(np.arange(n_runs)*config.num_episodes, np.min(summed_histories), np.max(summed_histories), colors='k', linestyles='dashed', label="Reset point")
    plt.legend()
    plt.title("Mean Social Welfare per Episode per Game")
    plt.xlabel("Episodes")
    plt.ylabel("Utility")
    plt.show()

    # for run in range(n_runs):
    #     sum_rewards_per_agent = np.sum(histories[run], axis=-1)
    #     for agent in range(num_agents):
    #         plt.plot(sum_rewards_per_agent[:, agent], label='Agent {}'.format(agents[agent].name))
    #         plt.legend()
    #     plt.show()

    print([agent.name for agent in agents])