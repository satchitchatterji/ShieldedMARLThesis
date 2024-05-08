import numpy as np
import wandb
import time
from tqdm import trange
from algos import BaseMARLAlgo

############################################ RUN EPISODE ############################################

def run_episode_MARL_algorithm(env, algo, max_cycles, ep=0):
    assert issubclass(type(algo), BaseMARLAlgo), "algo must be an instance of BaseMARLAlgo"
    reward_hist = {}
    observations, infos = env.reset()

    for step in trange(max_cycles, desc=f"Episode {ep}"):

        if len(env.agents) == 0:
            break

        actions = algo.act(observations)
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        algo.update_rewards(rewards, terminations, truncations)

        for agent in algo.agents.keys():
            if agent not in reward_hist:
                reward_hist[agent] = []
            reward_hist[agent].append(rewards[agent])
        
    update_dict = {}
    # mean reward per agent
    update_dict.update({f"mean_reward_{agent}": np.mean(reward_hist[agent]) for agent in reward_hist})
    # mean reward overall
    update_dict.update({"mean_reward": np.mean([np.mean(reward_hist[agent]) for agent in reward_hist])})
    wandb.log(update_dict)
    return reward_hist

def run_episode(env, agents, max_cycles, action_wrapper=None, ep=0):
    
    if issubclass(type(agents), BaseMARLAlgo):
        return run_episode_MARL_algorithm(env, agents, max_cycles, ep)

    reward_hist = {}
    observations, infos = env.reset()

    for step in trange(max_cycles, desc=f"Episode {ep}"):
    # for step in range(max_cycles):

        actions = {}
        for agent in env.agents:
            actions[agent] = action_wrapper(agents[agent].act(observations[agent]))
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # wandb.log({f"reward_{agent}": rewards[agent] for agent in env.agents})
        # wandb.log({"mean_reward": np.mean([rewards[agent] for agent in env.agents])})
        # wandb.log({f"safety_{agent}": agents[agent].debug_info["safety"]  for agent in env.agents if "safety" in agents[agent].debug_info})
        

        for agent in agents.keys():
            agents[agent].update_reward(rewards[agent], terminations[agent] or truncations[agent])

        for agent in agents.keys():
            if agent not in reward_hist:
                reward_hist[agent] = []
            reward_hist[agent].append(rewards[agent])

    update_dict = {}
    # mean reward per agent
    update_dict.update({f"mean_reward_{agent}": np.mean(reward_hist[agent]) for agent in reward_hist})
    # mean reward overall
    update_dict.update({"mean_reward": np.mean([np.mean(reward_hist[agent]) for agent in reward_hist])})
    wandb.log(update_dict)
    return reward_hist

############################################ EVALUATION ############################################

def eval_episode_MARL_algorithm(env, algo, max_cycles, ep=0):
    assert issubclass(type(algo), BaseMARLAlgo), "algo must be an instance of BaseMARLAlgo"
    reward_hist = {}
    observations, infos = env.reset()

    algo.eval(True)

    for step in range(max_cycles):

        if len(env.agents) == 0:
            break

        actions = algo.act(observations)
        
        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in algo.agents.keys():
            if agent not in reward_hist:
                reward_hist[agent] = []
            reward_hist[agent].append(rewards[agent])
        
    update_dict = {}
    # mean reward per agent
    update_dict.update({f"mean_reward_{agent}": np.mean(reward_hist[agent]) for agent in reward_hist})
    # mean reward overall
    update_dict.update({"mean_reward": np.mean([np.mean(reward_hist[agent]) for agent in reward_hist])})
    wandb.log(update_dict)

    algo.eval(False)

    return reward_hist

def eval_episode(env, agents, max_cycles, action_wrapper=None, ep=0):
    
    if issubclass(type(agents), BaseMARLAlgo):
        return eval_episode_MARL_algorithm(env, agents, max_cycles, ep)


    for agent in agents.keys():
        agents[agent].set_eval_mode(True)

    reward_hist = {}
    observations, infos = env.reset()

    for step in range(max_cycles):
    # for step in range(max_cycles):

        actions = {}
        for agent in env.agents:
            actions[agent] = action_wrapper(agents[agent].act(observations[agent]))
        
        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in agents.keys():
            if agent not in reward_hist:
                reward_hist[agent] = []
            reward_hist[agent].append(rewards[agent])

    update_dict = {}
    # mean reward per agent
    update_dict.update({f"mean_reward_{agent}": np.mean(reward_hist[agent]) for agent in reward_hist})
    # mean reward overall
    update_dict.update({"mean_reward": np.mean([np.mean(reward_hist[agent]) for agent in reward_hist])})
    wandb.log(update_dict)
    
    for agent in agents.keys():
        agents[agent].set_eval_mode(False)

    return reward_hist