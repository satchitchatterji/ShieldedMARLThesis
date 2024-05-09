import numpy as np
import wandb
import time
from tqdm import trange
from algos import BaseMARLAlgo

############################################ RUN EPISODE ############################################

def run_episode(env, algo, max_cycles, ep=0):
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


############################################ EVALUATION ############################################

def eval_episode(env, algo, max_cycles, ep=0, save_wandb=False):
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
    
    if save_wandb:
        update_dict = {}
        # mean reward per agent
        update_dict.update({f"eval_mean_reward_{agent}": np.mean(reward_hist[agent]) for agent in reward_hist})
        # mean reward overall
        update_dict.update({"eval_mean_reward": np.mean([np.mean(reward_hist[agent]) for agent in reward_hist])})
        wandb.log(update_dict)

    algo.eval(False)

    return reward_hist