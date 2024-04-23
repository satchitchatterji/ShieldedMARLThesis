import numpy as np
import wandb
import time
from tqdm import trange

def run_episode(env, agents, max_cycles, action_wrapper, ep):
    
    reward_hist = {}
    observations, infos = env.reset()

    for step in trange(max_cycles, desc=f"Episode {ep}"):

        actions = {}
        for agent in env.agents:
            actions[agent] = action_wrapper(agents[agent].act(observations[agent]))
    
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        wandb.log({f"reward_{agent}": rewards[agent] for agent in env.agents})
        wandb.log({"mean_reward": np.mean([rewards[agent] for agent in env.agents])})
        wandb.log({f"safety_{agent}": agents[agent].debug_info["safety"]  for agent in env.agents if "safety" in agents[agent].debug_info})

        for agent in env.agents:
            agents[agent].update_reward(rewards[agent], terminations[agent] or truncations[agent])

        for agent in env.agents:
            if agent not in reward_hist:
                reward_hist[agent] = []
            reward_hist[agent].append(rewards[agent])
            
        truncs = [terminations[agent] or truncations[agent] for agent in env.agents]
        if all(truncs):
            for agent in env.agents:
                agents[agent].reset()
            break

    return reward_hist

def eval_episode(env, agents, max_cycles, action_wrapper):
    for agent in agents:
        agents[agent].eval_mode = True
    observations, infos = env.reset()
    reward_hist = {}
    for step in range(max_cycles):

        actions = {}
        for agent in env.agents:
            actions[agent] = action_wrapper(agents[agent].act(observations[agent]))
    
        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in env.agents:
            if agent not in reward_hist:
                reward_hist[agent] = []
            reward_hist[agent].append(rewards[agent])

        truncs = [terminations[agent] or truncations[agent] for agent in env.agents]
        if all(truncs):
            for agent in env.agents:
                agents[agent].reset()
            break
    return reward_hist