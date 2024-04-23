import numpy as np

from pettingzoo.sisl import waterworld_v4
from dqn_shielded import DQNShielded
import matplotlib.pyplot as plt
from tqdm import trange

import os
system = os.name

import wandb
import time
cur_time = time.time()

raise NotImplementedError()

from action_wrappers import WaterworldActionWrapper
from sensor_wrappers import WaterworldSensorWrapper
from shield_selector import ShieldSelector

n_discrete_actions = 9
pursuer_max_accel = 0.01


max_cycles=500
env = waterworld_v4.parallel_env(render_mode=None, speed_features=False, pursuer_max_accel=pursuer_max_accel, max_cycles=max_cycles)
env.reset()

action_wrapper = WaterworldActionWrapper(n_discrete_actions, pursuer_max_accel, 1.0)
sensor_wrapper = WaterworldSensorWrapper(env, output_type="invert")

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

# - IDQL: Independent DQL
# - SIIDQL: Shield-Independent IDQL
# - SSIDQL: Shield-Sharing IDQL

# - PSDQL: Parameter-Sharing DQL
# - SIPSDQL: Shield-Independent PS-DQL
# - SSPSDQL: Shield-Sharing PS-DQL

agents = {}
training_style = "SSPSDQL"

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


reward_hist = {}
ep_rewards = []
for ep in range(0):
    wandb.init(project=f"{system}_waterworld", name=f"{training_style}_ep_{ep}_{cur_time}")
    ep_rewards_this = {}
    observations, infos = env.reset()

    for step in trange(max_cycles, desc=f"Episode {ep}"):

        actions = {}
        for agent in env.agents:
            actions[agent] = action_wrapper(agents[agent].act(observations[agent]))
    
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        wandb.log({f"reward_{agent}": rewards[agent] for agent in env.agents})
        wandb.log({f"safety_{agent}": agents[agent].debug_info["safety"]  for agent in env.agents if "safety" in agents[agent].debug_info})

        for agent in env.agents:
            agents[agent].update_reward(rewards[agent], terminations[agent] or truncations[agent])

        for agent in env.agents:
            if agent not in reward_hist:
                reward_hist[agent] = []
            if agent not in ep_rewards_this:
                ep_rewards_this[agent] = []
            reward_hist[agent].append(rewards[agent])
            ep_rewards_this[agent].append(rewards[agent])
            
        truncs = [terminations[agent] or truncations[agent] for agent in env.agents]
        if all(truncs):
            for agent in env.agents:
                agents[agent].reset()
            break

    ep_rewards.append({a:np.sum(ep_rewards_this[a]) for a in ep_rewards_this.keys()})
    ep_rewards[-1]["total"] = np.sum([np.sum(ep_rewards_this[a]) for a in ep_rewards_this.keys()])
    print(ep_rewards[-1])

    wandb.log({"total_reward": ep_rewards[-1]["total"]})
    wandb.finish()

env.close()

# for agent in reward_hist:
#     plt.plot(reward_hist[agent], label=agent)
# plt.legend()
# plt.show()

# for agent in ep_rewards[0]:
#     plt.plot([r[agent] for r in ep_rewards], label=agent)
# plt.plot([r["total"] for r in ep_rewards], label="total")
# plt.legend()
# plt.show()


# eval
import time

env = waterworld_v4.parallel_env(render_mode="human", speed_features=False, pursuer_max_accel=pursuer_max_accel, max_cycles=max_cycles)
observations, infos = env.reset()

for agent in agents:
    agents[agent].eval_mode = True

reward_hist = {}
for ep in range(1):
    print(f"Episode {ep}")
    observations, infos = env.reset()
    
    for step in range(max_cycles):
        # time.sleep(1)
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

# for agent in reward_hist:
#     plt.plot(reward_hist[agent], label=agent)
# plt.legend()
# plt.show()

print({a:np.sum(reward_hist[a]) for a in reward_hist.keys()})

env.close()