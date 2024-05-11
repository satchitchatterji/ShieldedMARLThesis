import os
import sys
import time
import pickle as pk

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import wandb

import action_wrappers
import sensor_wrappers
from shield_selector import ShieldSelector

from algos import *
from env_selection import ALL_ENVS, ALL_ENVS_ARGS
from config import config

from run_episode import run_episode, eval_episode

system = os.name
cur_time = time.time()

# set up environment
max_training_episodes = config.max_eps
max_cycles = config.max_cycles

total_cycles = max_training_episodes * max_cycles
print(f"Training for {max_training_episodes} episodes of {max_cycles} cycles each, totalling {total_cycles} cycles.")

env_creator_func = ALL_ENVS[config.env]
env_creator_args = ALL_ENVS_ARGS[config.env]
env_creator_args.update({"max_cycles": max_cycles})

env = env_creator_func(render_mode=None, **env_creator_args)
env.reset()

env_name = env.metadata["name"]
eval_every = config.eval_every
n_eval = config.n_eval

n_discrete_actions = env.action_space(env.possible_agents[0]).n
if hasattr(env.observation_space(env.possible_agents[0]), "shape") and len(env.observation_space(env.possible_agents[0]).shape) > 0: 
    observation_space = env.observation_space(env.possible_agents[0]).shape[0]  # for box spaces with shape
else: 
    observation_space = env.observation_space(env.possible_agents[0]).n         # for discrete spaces?

print(n_discrete_actions, observation_space)

action_wrapper = action_wrappers.IdentityActionWrapper(n_discrete_actions)
# sensor_wrapper = sensor_wrappers.IdentitySensorWrapper(env, observation_space)
sensor_wrapper = sensor_wrappers.MarkovStagHuntSensorWrapper(env, observation_space)

shield_selector = ShieldSelector(env_name=env_name, 
                                 n_actions=action_wrapper.num_actions, 
                                 n_sensors=sensor_wrapper.num_sensors,
                                 filename=config.shield_file
                                 )

sh_params = {
    "config_folder": shield_selector.base_dir,
    "num_sensors": sensor_wrapper.num_sensors,
    "num_actions": action_wrapper.num_actions,
    "differentiable": True,
    "shield_program": shield_selector.file,
    "observation_type": "ground truth",
    "get_sensor_value_ground_truth": sensor_wrapper,
}

# sh_params = None
ppo_params = ["update_timestep", "train_epochs", "gamma", "eps_clip", "lr_actor", "lr_critic"]
dqn_params = ["update_timestep", "train_epochs", "gamma", "buffer_size", "batch_size", "lr", "eps_decay", "eps_min", "tau", "target_update_type"]
extracted_ppo = {k: v for k, v in vars(config).items() if k in ppo_params}
extracted_dqn = {k: v for k, v in vars(config).items() if k in dqn_params}
all_algo_params = {k: v for k, v in vars(config).items() if k in ppo_params or k in dqn_params}
alpha = config.shield_alpha


############################################ ALGO SELECTION ############################################

algo_name = config.algo

if algo_name not in ALL_ALGORITHMS:
    raise ValueError(f"Algorithm '{algo_name}' not found. Available styles are {list(ALL_ALGORITHMS.keys())}")

algo = ALL_ALGORITHMS[algo_name](env=env, 
                                 observation_space=observation_space,
                                 n_discrete_actions=n_discrete_actions,
                                 action_wrapper=action_wrapper,
                                 sensor_wrapper=sensor_wrapper,
                                 sh_params=sh_params,
                                 algorithm_params=all_algo_params,
                                 alpha=alpha
                                 )

############################################ TRAINING ############################################

reward_hists = []
eval_hists = []
eval_episodes = []
wandb.init(project=f"{system}_{env_name}", name=f"{algo_name}_{cur_time}", config=vars(config))

ep=0
for _ in range(max_training_episodes):
    reward_hist = run_episode(env, algo, max_cycles, ep)
    reward_hists.append(reward_hist)

    if ep % eval_every == 0 or ep == max_training_episodes-1:
        eval_episodes.append(ep)
        eval_reward_hists = []
        for _ in range(n_eval):
            eval_reward_hist = eval_episode(env, algo, max_cycles)
            eval_reward_hists.append(eval_reward_hist)
        eval_hists.append(eval_reward_hists)

        algo.save(f"models/{env_name}/{algo_name}_{cur_time}/ep{ep}")

    ep+=1

wandb.finish()
env.close()

############################################ POST TRAINING ############################################

# for r, reward_hist in enumerate(reward_hists):
#     print(f"Episode {r} : ", end="")    
#     print({a:np.sum(reward_hist[a]) for a in reward_hist.keys()})
#     print("Total reward: ", np.sum([np.sum(reward_hist[a]) for a in reward_hist.keys()]))

# save hists as pickle
if not os.path.exists(f"histories/{env_name}/{algo_name}/{cur_time}_train.pk"):
    os.makedirs(f"histories/{env_name}/{algo_name}", exist_ok=True)
with open(f"histories/{env_name}/{algo_name}/{cur_time}_train.pk", "wb") as f:
    pk.dump(reward_hists, f)

if not os.path.exists(f"histories/{env_name}/{algo_name}/{cur_time}_eval.pk"):
    os.makedirs(f"histories/{env_name}/{algo_name}", exist_ok=True)
with open(f"histories/{env_name}/{algo_name}/{cur_time}_eval.pk", "wb") as f:
    pk.dump(eval_hists, f)

print("Training complete. Fileref:", cur_time)
# safe config dict to json file
import json
if not os.path.exists(f"configs/{env_name}/{algo_name}/{cur_time}.json"):
    os.makedirs(f"configs/{env_name}/{algo_name}", exist_ok=True)
with open(f"configs/{env_name}/{algo_name}/{cur_time}.json", "w") as f:
    s = json.dumps(vars(config), indent=4)
    f.write(s)

# plot mean rewards per episode
agent_rewards = [[np.mean(reward_hist[agent]) for reward_hist in reward_hists] for agent in reward_hists[0].keys()]
for a, agent in enumerate(algo.agents.keys()):
    plt.plot(agent_rewards[a], label=agent)

plt.plot([np.mean([np.mean(reward_hist[agent]) for agent in reward_hist.keys()]) for reward_hist in reward_hists], label="mean", color="black", linestyle="--")

plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title(f"{algo_name} on {env_name} (training)")

plt.legend()
if not os.path.exists(f"plots/{env_name}/{algo_name}"):
    os.makedirs(f"plots/{env_name}/{algo_name}")
plt.grid(True)
plt.savefig(f"plots/{env_name}/{algo_name}/{cur_time}_train.png")
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
    # TODO: Breaks when max_training_episodes is not divisible by eval_every
    plt.errorbar(eval_episodes, eval_means[agent], yerr=eval_stds[agent], label=f"{agent} mean", capsize=5, marker="x")

plt.errorbar(eval_episodes, eval_means["mean"], yerr=eval_stds["mean"], label="mean", color="black", linestyle="--", capsize=5, marker="x")

# plt.xticks(eval_episodes)
plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title(f"{algo_name} on {env_name} (evaluation)")
plt.grid(True)
plt.legend()
plt.savefig(f"plots/{env_name}/{algo_name}/{cur_time}_eval.png")

# plt.show()
exit()

############################################ OFFLINE EVALUATION ############################################
# set up environment
env = env_creator_func(render_mode=None, **env_creator_args)
env.reset()
# evaluation episodes
del algo
algo = ALL_ALGORITHMS[algo_name](env=env, 
                                 observation_space=observation_space,
                                 n_discrete_actions=n_discrete_actions,
                                 action_wrapper=action_wrapper,
                                 sensor_wrapper=sensor_wrapper,
                                 sh_params=sh_params,
                                 algorithm_params=all_algo_params,
                                 alpha=alpha
                                 )
algo.load(f"models/{env_name}/{algo_name}_{cur_time}/ep{max_training_episodes-1}")
# algo.update_env(env)
reward_hists = []
for ep in range(5):
    reward_hist = eval_episode(env, algo, max_cycles)
    reward_hists.append(reward_hist)

for reward_hist in reward_hists:
    print({a:np.sum(reward_hist[a]) for a in reward_hist.keys()})

env.close() 