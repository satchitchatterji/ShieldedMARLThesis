import os
import datetime
import pickle as pk

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
import json
import pprint

import action_wrappers
import sensor_wrappers
from shield_selector import ShieldSelector

from algos import *
from env_selection import ALL_ENVS, ALL_ENVS_ARGS
from config import config

from run_episode import run_episode, eval_episode

np.random.seed(config.seed)
torch.manual_seed(config.seed)


system = os.name
# cur_time = time.time()
now = datetime.datetime.now()
cur_time = now.strftime("%Y-%m-%d_%H%M%S")

# set up environment
max_training_episodes = config.max_eps
max_cycles = config.max_cycles

env_creator_func = ALL_ENVS[config.env]
env_creator_args = ALL_ENVS_ARGS[config.env]
env_creator_args.update({"max_cycles": max_cycles})

env = env_creator_func(render_mode="human", **env_creator_args)
env.reset()

n_eval = config.n_eval
env_name = env.metadata["name"]
algo_name = config.algo

base_folder = f"models/{env_name}"
subfolders = [x for x in os.listdir(base_folder) if x.startswith(algo_name)]
latest_subfolder = max(subfolders)
print("Latest subfolder", latest_subfolder)
folder = f"models/{env_name}/{latest_subfolder}"
all_eps_in_folder = [int(x.replace("ep","")) for x in os.listdir(folder)]
max_eps = max(all_eps_in_folder)
print("Loading model from", max_eps)

config_file = f"configs/{config.env}/{algo_name}/{latest_subfolder.replace(algo_name+'_','')}.json"

with open(config_file, "r") as f:
    print(f"Loading config from {config_file}")
    config.__dict__ = json.load(f)
    pprint.pprint(config.__dict__)

config.eval_policy = "greedy"
config.shield_version = 1

del env

max_training_episodes = config.max_eps
max_cycles = config.max_cycles

total_cycles = max_training_episodes * max_cycles

env_creator_func = ALL_ENVS[config.env]
env_creator_args = ALL_ENVS_ARGS[config.env]
env_creator_args.update({"max_cycles": max_cycles})

env = env_creator_func(render_mode="human", **env_creator_args)
env.reset()


n_discrete_actions = env.action_space(env.possible_agents[0]).n
if hasattr(env.observation_space(env.possible_agents[0]), "shape") and len(env.observation_space(env.possible_agents[0]).shape) > 0: 
    observation_space = env.observation_space(env.possible_agents[0]).shape[0]  # for box spaces with shape
else: 
    observation_space = env.observation_space(env.possible_agents[0]).n         # for discrete spaces?

print(f"Observation space: {observation_space}, Action space: {n_discrete_actions}")
action_wrapper = action_wrappers.IdentityActionWrapper(n_discrete_actions)
# sensor_wrapper = sensor_wrappers.IdentitySensorWrapper(env, observation_space)
sensor_wrapper = sensor_wrappers.get_wrapper(env_name)(env, observation_space)

shield_selector = ShieldSelector(env_name=env_name, 
                                 n_actions=action_wrapper.num_actions, 
                                 n_sensors=sensor_wrapper.num_sensors,
                                 filename=config.shield_file,
                                 version=config.shield_version
                                 )

sh_params = {
    "config_folder": shield_selector.base_dir,
    "num_sensors": sensor_wrapper.num_sensors,
    "num_actions": action_wrapper.num_actions,
    "differentiable": config.shield_diff,
    "vsrl_eps": config.shield_eps,
    "shield_program": shield_selector.file,
    "observation_type": "ground truth",
    "get_sensor_value_ground_truth": sensor_wrapper,
}
# sh_params = None

# print(sh_params)

ppo_params = ["update_timestep", "train_epochs", "gamma", "eps_clip", "lr_actor", "lr_critic"]
dqn_params = ["update_timestep", "train_epochs", "gamma", "buffer_size", "batch_size", "lr", "eps_decay", "eps_min", "tau", "target_update_type", "explore_policy", "eval_policy"]
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
                                 alpha=alpha,
                                 shielded_ratio=config.shielded_ratio
                                 )


algo.load(folder + "/" + f"ep{max_eps}")

############################################ SAFETY CALC ############################################

# safety_calc = SafetyCalculator(sh_params)
safety_calc = None 

############################################ TRAINING ############################################

reward_hists = []
eval_hists = []
eval_safeties = []
eval_episodes = []

ep=0

for _ in range(1):
    eval_episodes.append(ep)
    eval_reward_hists = []
    eval_safety_hists = []
    for _ in range(n_eval):
        eval_reward_hist, eval_safety_hist = eval_episode(env, algo, max_cycles, save_wandb=False)
        eval_reward_hists.append(eval_reward_hist)
        eval_safety_hists.append(eval_safety_hist)
    eval_hists.append(eval_reward_hists)
    eval_safeties.append(eval_safety_hists)

    ep+=1

env.close()

# print(eval_hists)