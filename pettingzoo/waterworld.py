from pettingzoo.sisl import waterworld_v4

env = waterworld_v4.parallel_env(render_mode="human", speed_features = False)
observations, infos = env.reset()
print(observations["pursuer_1"].shape)

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()