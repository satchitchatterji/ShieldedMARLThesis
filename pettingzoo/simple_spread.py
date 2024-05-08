import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import wandb

# from action_wrappers import WaterworldActionWrapper
# from sensor_wrappers import WaterworldSensorWrapper
from shield_selector import ShieldSelector

from pettingzoo.mpe import simple_spread_v3

from algos import *
from run_episode import run_episode, eval_episode

# accept cmd line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--training_style", 
                    type=str, 
                    default="IQL", 
                    help="Training style to use. Options are IQL, PSIQL, IPPO, MAPPO, ACSPPO, SIQL, SPSIQL, SIPPO, SMAPPO, SACSPPO"
                    )

system = os.name
cur_time = time.time()

# set up environment
max_training_episodes=10
max_cycles=25
env = simple_spread_v3.parallel_env(render_mode=None, N=3, local_ratio=0.5, max_cycles=max_cycles, continuous_actions=False)

env_name = env.metadata["name"]
eval_every = 5
n_eval = 10

env.reset()
n_discrete_actions = env.action_space(env.possible_agents[0]).n
observation_space = env.observation_space(env.possible_agents[0]).shape[0]

# action_wrapper = WaterworldActionWrapper(n_discrete_actions, pursuer_max_accel, 1.0)
action_wrapper = lambda x: x
sensor_wrapper = lambda x: x
# sensor_wrapper = WaterworldSensorWrapper(env, output_type="reduce_to_8")
# shield_file = ShieldSelector(env_name="waterworld", 
#                             n_actions=action_wrapper.n_actions, 
#                             n_sensors=sensor_wrapper.num_sensors)
# sh_params = {
#     "config_folder": ".",
#     "num_sensors": sensor_wrapper.num_sensors,
#     "num_actions": n_discrete_actions,
#     "differentiable": True,
#     "shield_program": shield_file.file,
#     "observation_type": "ground truth",
#     "get_sensor_value_ground_truth": sensor_wrapper,
# }

# reference means: vals
# IQL:     −155.81 ± 1.50
# IPPO:    −149.89 ± 2.91
# MAPPO:   −149.26 ± 0.94
# PSIQL:   −141.87 ± 1.68
# ACSPPO:  −132.46 ± 3.54 or −128.64 ± 2.83 (without identity?)


sh_params = None
alpha = 1.0
default_ppo_params = {
    "update_timestep": int(max_cycles/4),      # update policy every n timesteps # TODO: check this
    "K_epochs": 50,               # update policy for K epochs in one PPO update
    "eps_clip": 0.2,          # clip parameter for PPO
    "gamma": 0.99,            # discount factor
    "lr_actor": 0.001,     # learning rate for actor network
    "lr_critic": 0.001,       # learning rate for critic network
}

shielded_algos = {
    "SIQL": SIQL, 
    "SPSIQL": SPSIQL, 
    "SIPPO": SIPPO, 
    "SMAPPO": SMAPPO, 
    "SACSPPO": SACSPPO
}

unshielded_algos = {
    "IQL": IQL, 
    "PSIQL": PSIQL, 
    "IPPO": IPPO, 
    "MAPPO": MAPPO, 
    "ACSPPO": ACSPPO
}

availible = {**shielded_algos, **unshielded_algos}

training_style = parser.parse_args().training_style

if training_style not in availible:
    raise ValueError(f"Training style {training_style} not found. Available styles are {availible.keys()}")

algo = availible[training_style](env=env, 
                                 observation_space=observation_space,
                                 n_discrete_actions=n_discrete_actions,
                                 action_wrapper=action_wrapper,
                                 sensor_wrapper=sensor_wrapper,
                                 sh_params=sh_params,
                                 algorithm_params=default_ppo_params,
                                 alpha=alpha
                                 )

# training episodes
reward_hists = []
eval_hists = []
wandb.init(project=f"{system}_{env_name}", name=f"{training_style}_{cur_time}")
ep=0
for _ in trange(max_training_episodes):
    reward_hist = run_episode(env, algo, max_cycles, action_wrapper, ep)
    reward_hists.append(reward_hist)

    if ep % eval_every == 0 or ep == max_training_episodes-1:
        eval_reward_hists = []
        for _ in range(n_eval):
            eval_reward_hist = eval_episode(env, algo, max_cycles, action_wrapper)
            eval_reward_hists.append(eval_reward_hist)
        eval_hists.append(eval_reward_hists)

    ep+=1
wandb.finish()
env.close()

# for r, reward_hist in enumerate(reward_hists):
#     print(f"Episode {r} : ", end="")    
#     print({a:np.sum(reward_hist[a]) for a in reward_hist.keys()})
#     print("Total reward: ", np.sum([np.sum(reward_hist[a]) for a in reward_hist.keys()]))


# plot mean rewards per episode
agent_rewards = [[np.mean(reward_hist[agent]) for reward_hist in reward_hists] for agent in reward_hists[0].keys()]
for a, agent in enumerate(algo.agents.keys()):
    plt.plot(agent_rewards[a], label=agent)

plt.plot([np.mean([np.mean(reward_hist[agent]) for agent in reward_hist.keys()]) for reward_hist in reward_hists], label="mean", color="black", linestyle="--")

plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title(f"{training_style} on {env_name} (training)")

plt.legend()
if not os.path.exists(f"plots/{env_name}"):
    os.makedirs(f"plots/{env_name}")
plt.grid(True)
plt.savefig(f"plots/{env_name}/{training_style}_{cur_time}_train.png")
# plt.show()
plt.clf()
# compute eval means and stds
eval_means = {}
eval_stds = {}

for a, agent in enumerate(algo.agents.keys()):
    eval_means[agent] = []
    eval_stds[agent] = []
    for eval_hist in eval_hists:
        eval_means[agent].append(np.mean([np.mean(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))
        eval_stds[agent].append(np.std([np.mean(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))

# compute overall mean and std
eval_means["mean"] = []
eval_stds["mean"] = []
for eval_hist in eval_hists:
    eval_means["mean"].append(np.mean([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
    eval_stds["mean"].append(np.std([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))

# plot eval info
for a, agent in enumerate(algo.agents.keys()):
    plt.errorbar(range(0, max_training_episodes+1, eval_every), eval_means[agent], yerr=eval_stds[agent], label=f"{agent} mean")

plt.errorbar(range(0, max_training_episodes+1, eval_every), eval_means["mean"], yerr=eval_stds["mean"], label="mean", color="black", linestyle="--")

plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title(f"{training_style} on {env_name} (evaluation)")
plt.grid(True)
plt.legend()
plt.savefig(f"plots/{env_name}/{training_style}_{cur_time}_eval.png")

# plt.show()

exit()

############################################ EVALUATION ############################################
# set up environment
env = parallel_stag_hunt.parallel_env(render_mode="human", max_cycles=max_cycles)
observations, infos = env.reset()
# evaluation episodes
reward_hists = []
for ep in range(1):
    print(f"Episode {ep}")
    reward_hist = eval_episode(env, agents, max_cycles, action_wrapper)
    reward_hists.append(reward_hist)

for reward_hist in reward_hists:
    print({a:np.sum(reward_hist[a]) for a in reward_hist.keys()})

env.close() 