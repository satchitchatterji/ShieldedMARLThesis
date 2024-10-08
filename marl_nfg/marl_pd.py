import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from tqdm import trange

from prisoners_dilemma_ma import PrisonersDilemmaMAEnv as make_env

from discretize import *
from config import config

from always_c_agent import CAgent
from always_d_agent import DAgent
from random_agent import RandomAgent
from tittat_agent import TitForTatAgent
from pd_q_agent import PDQAgent
from pd_sarsa_agent import PDSARSAAgent
from pd_deepsarsa_agent import PDDeepSARSAAgent
# from pd_dqn_agent import PDDQNAgent
from pd_dqns_agent import PDDQNShieldedAgent


l_reward_factor = 0

# def compute_semantic_reward(env, agents, rewards):
#     # input: rewards is a 2D array of shape (num_agents, num_agents)
#     # output: semantic_rewards is a 2D array of shape (num_agents, num_agents)
#     # semantic_rewards[i][j] is the semantic reward of agent i against agent j
#     # L_C=-\sum_{x,y,t}\log(P(x,y,t)|C(y,x,t-1))
#     # L_C=-\sum_{x,y,t} P(x,y,t)\log(C(y,x,t-1))

#     for agent in agents:
#         if agent.learning:
#             pass

# def compute_psl_reward(env, agents, rewards):
#     # input: rewards is a 2D array of shape (num_agents, num_agents)
#     # output: psl_rewards is a 2D array of shape (num_agents, num_agents)
#     # psl_rewards[i][j] is the psl reward of agent i against agent j
#     # For agents x and y, L_psl = \min(1-C(y,x,t-1)+C(x,y,t),1)

#     psl_rewards = np.zeros(rewards.shape)
#     for a, agent in enumerate(agents):
#         agent_reward = rewards[a]
#         if not agent.learning:
#             psl_rewards[a] = np.zeros(rewards.shape[0])
#         else:
#             for o, opponent in enumerate(agents):
#                 if o != a:
#                     opponent_prev_action = 1 - opponent.prev_actions[a] # 1 if opponent cooperated, 0 if opponent defected
#                     agent_action = 1
#                     psl_rewards[a][o] = min(1 - opponent_prev_action + agent_action, 1)
                    
def compute_l_reward(env, agents, rewards):
    # input: rewards is a 2D array of shape (num_agents, num_agents)
    # output: l_rewards is a 2D array of shape (num_agents, num_agents)
    # l_rewards[i][j] is the l reward of agent i against agent j
    l_rewards = np.zeros(rewards.shape)
    # reward_table = [1,1] # C or D
    # reward_table = [1,-1] # C and not D
    # reward_table = [-1,1] # C and not D
    reward_table = [-1,-1] # not C and not D
    for a, agent in enumerate(agents):
        if not agent.learning:
            l_rewards[a] = np.zeros(rewards.shape[0])
        else:
            for o, opponent in enumerate(agents):
                if o != a:
                    agent_action = env.state_history[-1][a][o]
                    # print(agent_action)
                    l_rewards[a][o] = reward_table[agent_action]
                    
    return l_rewards


def run_episode(env, agents, max_steps, ep_num=None):
    observation = env.reset()[0]
    total_agent_rewards = np.zeros((len(agents), len(agents)))

    for agent in agents:
        agent.begin_episode()
    
    for _ in trange(max_steps, desc=f'Episode {ep_num} / {config.num_episodes}'):
        # env.render()
        actions = []
        for i, agent in enumerate(agents):
            actions.append(agent.act(np.array(observation).T[i]))
        
        observation, rewards, terminated, truncated, info = env.step(actions)
        l_rewards = compute_l_reward(env, agents, rewards)
        rewards += l_reward_factor*l_rewards
        
        for i, agent in enumerate(agents):
            # Optimize individual reward
            agent.update_reward(rewards[i], terminated or truncated)

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
        rewards = run_fn(env, agents, env.spec.max_episode_steps, ep_num=e)
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
    all_state_histories = []
    all_rewards_histories = []
    all_safe_loss_histories = {}
    all_base_loss_histories = {}
    for i in range(n_runs):
        
        sh_params = {
            "config_folder": ".",
            "num_sensors": 4,
            "num_actions": 2,
            "differentiable": True,
            "shield_program": "dqn_shield.pl",
            "observation_type": "ground truth",
        }

        sh_params2 = {
            "config_folder": ".",
            "num_sensors": 6,
            "num_actions": 2,
            "differentiable": True,
            "shield_program": "dqn_shield2.pl",
            "observation_type": "ground truth",
        }

        agents = [
            # RandomAgent(num_states=2, num_actions=2)
            # CAgent(num_states=2, num_actions=2),
            # DAgent(num_states=2, num_actions=2),
            # TitForTatAgent(num_states=2, num_actions=2),
            # PDQAgent(num_states=2, num_actions=2),
            # PDQAgent(num_states=2, num_actions=2),
            # PDSARSAAgent(num_states=2, num_actions=2),
            # PDDeepSARSAAgent(num_states=2, num_actions=2),
            # PDDeepSARSAAgent(num_states=2, num_actions=2),
            # PDDQNAgent(num_states=2, num_actions=2),
            PDDQNShieldedAgent(num_states=2, num_actions=2, shield_params=sh_params2),
            PDDQNShieldedAgent(num_states=2, num_actions=2, shield_params=sh_params2),
            PDDQNShieldedAgent(num_states=2, num_actions=2, shield_params=sh_params2)
        ]
        # agents.append(PDDQNShieldedAgent(num_states=2, num_actions=2, shield=agents[0].shield))

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
        if agents[0].learning:
            print('Run {}: Agent 0 params:'.format(i))
            print(agents[0].get_params())
            # epsilon_histories += agents[0].epsilon_histories1
        
        # determinisitcs.append(determinisitc)
        
        all_state_histories.append(env.state_history)
        all_rewards_histories.append(env.rewards_history)

        for a, agent in enumerate(agents):
            if agent.name == "DQNShielded":
                safety = []
                base_loss = []
                for iter in agent.debug_info_history:
                    safety.append(iter['safety'].squeeze(0).item())

                for iter in agent.loss_info:
                    # safety.append(iter['safety_loss'].squeeze(0).item())
                    base_loss.append(iter['base_loss'].squeeze(0).item())

                if agent.name not in all_safe_loss_histories:
                    all_safe_loss_histories[agent.name+str(a)] = list()
                    all_base_loss_histories[agent.name+str(a)] = list()

                all_safe_loss_histories[agent.name+str(a)] += safety
                all_base_loss_histories[agent.name+str(a)] += base_loss

    # plot safety
    fig, ax1 = plt.subplots()

    for agent, loss in all_safe_loss_histories.items():
        ax1.plot(loss, label=agent+" safety")
    # ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    for agent, loss in all_base_loss_histories.items():
        ax2.plot(loss, linestyle="dashed", label=agent+" base")

    plt.title("Training stuff")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)
    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("Safety loss")
    ax2.set_ylabel("Base loss")
    # ax1.vlines(np.arange(n_runs+1)*config.num_episodes*env.spec.max_episode_steps, 0, 1, colors='k', linestyles='dashed', label="Reset point")
    plt.show()
    # exit()


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
    # exit()
    
    # # plot epsilon
    
    # if agents[0].name == 'Q-Learning':
    #     plt.plot(epsilon_histories, label='epsilon', color='g')
    #     plt.show()

    # # plot sum of all agents' rewards per run
    sum_rewards_per_agent_all_games = (np.sum(histories, axis=-1)/env.spec.max_episode_steps)
    summed_histories = np.sum(sum_rewards_per_agent_all_games, axis=-1)
    # plt.plot(summed_histories)
    # plt.vlines(np.arange(n_runs)*config.num_episodes, np.min(summed_histories), np.max(summed_histories), colors='k', linestyles='dashed', label="Reset point")
    # plt.legend()
    # plt.title("Mean Social Welfare for All Games per Episode")
    # plt.xlabel("Episodes")
    # plt.ylabel("Utility")
    # plt.show()

    # # n choose 2 games
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

    # plot heatmaps of rewards for each run
    fig, axs = plt.subplots(1, n_runs, figsize=(10, 5))

    all_state_histories = np.array(all_state_histories)    
    for run in range(n_runs):
        this_run_state_history = all_state_histories[run][-10:, :, :]
        print(this_run_state_history[-1])
        mean_actions = this_run_state_history.mean(axis=0)
        # green to red heatmap
        sns.heatmap(mean_actions, ax=axs[run], annot=True, cmap="RdYlGn_r")
        axs[run].set_title("Run {}".format(run))
        axs[run].set_ylabel("Agent")
        axs[run].set_xlabel("Opponent")
        axs[run].set_xticklabels([agent.name for agent in agents])
        axs[run].set_yticklabels([agent.name for agent in agents])
    fig.tight_layout()
    plt.title("Mean Action per Agent per Episode")
    plt.show()    

    fig, axs = plt.subplots(1, n_runs, figsize=(10, 5))

    all_rewards_histories = np.array(all_rewards_histories)    
    for run in range(n_runs):
        this_run_rewards_history = all_rewards_histories[run][-10:, :, :]
        print(this_run_rewards_history[-1])
        mean_rewards = this_run_rewards_history.mean(axis=0)
        # sns heatmaps

        sns.heatmap(mean_rewards, annot=True, fmt=".2f", ax=axs[run], cmap="RdYlGn", center=None)
        axs[run].set_title("Run {}".format(run))
        axs[run].set_ylabel("Agent")
        axs[run].set_xlabel("Opponent")
        axs[run].set_xticklabels([agent.name for agent in agents])
        axs[run].set_yticklabels([agent.name for agent in agents])
    fig.tight_layout()
    plt.title("Mean Reward per Agent per Episode")
    plt.show()