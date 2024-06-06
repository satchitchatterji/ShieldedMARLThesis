import centipede_env

env = centipede_env.raw_env(render_mode="human")

observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(rewards)
env.close()