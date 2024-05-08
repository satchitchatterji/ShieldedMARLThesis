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

sys.path.append("../grid_envs")
import parallel_stag_hunt
from algos import *
from run_episode import run_episode, eval_episode

system = os.name
cur_time = time.time()

max_cycles=100
env = parallel_stag_hunt.parallel_env(render_mode=None, max_cycles=max_cycles)
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
sh_params = None
alpha = 1.0
default_ppo_params = {
    "update_timestep": int(max_cycles/4),      # update policy every n timesteps # TODO: check this
    "K_epochs": 50,               # update policy for K epochs in one PPO update
    "eps_clip": 0.2,          # clip parameter for PPO
    "gamma": 0.99,            # discount factor
    "lr_actor": 0.01,     # learning rate for actor network
    "lr_critic": 0.01,       # learning rate for critic network
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

training_style = "MAPPO"

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
wandb.init(project=f"{system}_stag_hunt", name=f"{training_style}_{cur_time}")
ep=0
while True:
    reward_hist = run_episode(env, algo, max_cycles, action_wrapper, ep)
    reward_hists.append(reward_hist)
    ep+=1
wandb.finish()
env.close()

# for r, reward_hist in enumerate(reward_hists):
#     print(f"Episode {r} : ", end="")    
#     print({a:np.sum(reward_hist[a]) for a in reward_hist.keys()})
#     print("Total reward: ", np.sum([np.sum(reward_hist[a]) for a in reward_hist.keys()]))


# plot mean rewards per episode
agent_rewards = [[np.sum(reward_hist[agent]) for reward_hist in reward_hists] for agent in reward_hists[0].keys()]
for a, agent in enumerate(algo.agents.keys()):
    plt.plot(agent_rewards[a], label=agent)

plt.plot([np.mean([np.sum(reward_hist[agent]) for agent in reward_hist.keys()]) for reward_hist in reward_hists], label="Mean Reward", color="black", linestyle="--")

plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title(f"{training_style} on Stag Hunt")

plt.legend()
plt.show()

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