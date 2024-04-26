import numpy as np

from pettingzoo.sisl import waterworld_v4
from dqn_shielded import DQNShielded
from ppo_shielded import PPOShielded
import matplotlib.pyplot as plt
from tqdm import trange

import os
system = os.name

import wandb
import time
cur_time = time.time()

from action_wrappers import WaterworldActionWrapper
from sensor_wrappers import WaterworldSensorWrapper
from shield_selector import ShieldSelector

from run_episode import run_episode, eval_episode

n_discrete_actions = 9
pursuer_max_accel = 0.01
n_sensors = 32
max_cycles=500
env = waterworld_v4.parallel_env(render_mode=None, n_sensors=n_sensors, speed_features=False, pursuer_max_accel=pursuer_max_accel, max_cycles=max_cycles)
env.reset()

action_wrapper = WaterworldActionWrapper(n_discrete_actions, pursuer_max_accel, 1.0)
sensor_wrapper = WaterworldSensorWrapper(env, output_type="reduce_to_8")
shield_file = ShieldSelector(env_name="waterworld", 
                            n_actions=action_wrapper.n_actions, 
                            n_sensors=sensor_wrapper.num_sensors)

sh_params = {
    "config_folder": ".",
    "num_sensors": sensor_wrapper.num_sensors,
    "num_actions": action_wrapper.n_actions,
    "differentiable": True,
    "shield_program": shield_file.file,
    "observation_type": "ground truth",
    "get_sensor_value_ground_truth": sensor_wrapper,
}

default_ppo_params = {
    "update_timestep": int(max_cycles/3),      # update policy every n timesteps # TODO: check this
    "K_epochs": 50,               # update policy for K epochs in one PPO update
    "eps_clip": 0.2,          # clip parameter for PPO
    "gamma": 0.99,            # discount factor
    "lr_actor": 0.0003,     # learning rate for actor network
    "lr_critic": 0.001,       # learning rate for critic network
}
# - IDQL: Independent DQL
# - SIIDQL: Shield-Independent IDQL
# - SSIDQL: Shield-Sharing IDQL

# - PSDQL: Parameter-Sharing DQL
# - SIPSDQL: Shield-Independent PS-DQL
# - SSPSDQL: Shield-Sharing PS-DQL

# TODO: The shield has no learning elements so all the SS variants are the same as the SI variants

agents = {}
training_style = "SACSPPO"

############################################ DQL ############################################

if training_style == "IDQL":
    for agent in env.agents:
        agents[agent] = DQNShielded(env.observation_spaces[agent].shape[0], n_discrete_actions)

elif training_style == "SIIDQL":
    for agent in env.agents:
        agents[agent] = DQNShielded(env.observation_spaces[agent].shape[0], n_discrete_actions, shield_params=sh_params)

elif training_style == "SSIDQL":
    agents[env.agents[0]] = DQNShielded(env.observation_spaces[env.agents[0]].shape[0], n_discrete_actions, shield_params=sh_params)
    for a, agent in enumerate(env.agents):
        if a != 0:
            agents[agent] = DQNShielded(env.observation_spaces[agent].shape[0], n_discrete_actions, shield=agents[env.agents[0]].shield)

elif training_style == "PSDQL":
    agents[env.agents[0]] = DQNShielded(env.observation_spaces[env.agents[0]].shape[0], n_discrete_actions)
    for a, agent in enumerate(env.agents):
        if a != 0:
            agents[agent] = DQNShielded(env.observation_spaces[agent].shape[0], n_discrete_actions, func_approx=agents[env.agents[0]].func_approx)

elif training_style == "SIPSDQL":
    agents[env.agents[0]] = DQNShielded(env.observation_spaces[env.agents[0]].shape[0], n_discrete_actions, shield_params=sh_params)
    for a, agent in enumerate(env.agents):
        if a != 0:
            agents[agent] = DQNShielded(env.observation_spaces[agent].shape[0], n_discrete_actions, func_approx=agents[env.agents[0]].func_approx, shield_params=sh_params)

elif training_style == "SSPSDQL":
    agents[env.agents[0]] = DQNShielded(env.observation_spaces[env.agents[0]].shape[0], n_discrete_actions, shield_params=sh_params)
    for a, agent in enumerate(env.agents):
        if a != 0:
            agents[agent] = DQNShielded(env.observation_spaces[agent].shape[0], n_discrete_actions, func_approx=agents[env.agents[0]].func_approx, shield=agents[env.agents[0]].shield)

############################################ PPO ############################################

elif training_style == "IPPO":
    for agent in env.agents:
        agents[agent] = PPOShielded(state_dim=env.observation_spaces[agent].shape[0], 
                                    action_dim=n_discrete_actions, 
                                    policy_kw_args={"get_sensor_value_ground_truth":sensor_wrapper},
                                    **default_ppo_params)

elif training_style == "SIPPO":
    for agent in env.agents:
        agents[agent] = PPOShielded(state_dim=env.observation_spaces[agent].shape[0], 
                                    action_dim=n_discrete_actions, 
                                    alpha=1, 
                                    policy_safety_params=sh_params,
                                    policy_kw_args={"shield_params":sh_params, "get_sensor_value_ground_truth":sensor_wrapper},
                                    **default_ppo_params)

elif training_style == "SSIPPO":
    agents[env.agents[0]] = PPOShielded(state_dim=env.observation_spaces[env.agents[0]].shape[0], 
                                    action_dim=n_discrete_actions, 
                                    alpha=1, 
                                    policy_safety_params=sh_params,
                                    policy_kw_args={"shield_params":sh_params, "get_sensor_value_ground_truth":sensor_wrapper},
                                    **default_ppo_params)
    for a, agent in enumerate(env.agents):
        if a != 0:
            agents[agent] = PPOShielded(state_dim=env.observation_spaces[agent].shape[0], 
                                    action_dim=n_discrete_actions, 
                                    alpha=1,
                                    policy_safety_params=sh_params,
                                    policy_kw_args={"shield": agents[env.agents[0]].policy.shield, "get_sensor_value_ground_truth":sensor_wrapper},
                                    **default_ppo_params)

elif training_style == "ACSPPO":
    for agent in env.agents:
        agents[agent] = PPOShielded(state_dim=env.observation_spaces[agent].shape[0], 
                                    action_dim=n_discrete_actions, 
                                    alpha=1, 
                                    policy_safety_params=sh_params,
                                    policy_kw_args={"shield_params":sh_params, "get_sensor_value_ground_truth":sensor_wrapper},
                                    **default_ppo_params)
    for a, agent in enumerate(env.agents):
        if a != 0:
            agents[agent].set_policy(agents[env.agents[0]].policy)

elif training_style == "SACSPPO":
    agents[env.agents[0]] = PPOShielded(state_dim=env.observation_spaces[env.agents[0]].shape[0], 
                                    action_dim=n_discrete_actions, 
                                    alpha=1, 
                                    policy_safety_params=sh_params,
                                    policy_kw_args={"shield_params":sh_params, "get_sensor_value_ground_truth":sensor_wrapper},
                                    **default_ppo_params)
    for a, agent in enumerate(env.agents):
        if a != 0:
            agents[agent] = PPOShielded(state_dim=env.observation_spaces[agent].shape[0], 
                                    action_dim=n_discrete_actions, 
                                    alpha=1,
                                    policy_safety_params=sh_params,
                                    policy_kw_args={"shield": agents[env.agents[0]].policy.shield, "get_sensor_value_ground_truth":sensor_wrapper},
                                    **default_ppo_params)
            agents[agent].set_policy(agents[env.agents[0]].policy)

# training episodes
reward_hists = []
for ep in range(10):
    wandb.init(project=f"{system}_waterworld", name=f"{training_style}_ep_{ep}_{cur_time}")
    reward_hist = run_episode(env, agents, max_cycles, action_wrapper, ep)
    reward_hists.append(reward_hist)
    wandb.finish()
env.close()
print(reward_hists)

# set up environment
env = waterworld_v4.parallel_env(render_mode="human", n_sensors=n_sensors, speed_features=False, pursuer_max_accel=pursuer_max_accel, max_cycles=max_cycles)
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