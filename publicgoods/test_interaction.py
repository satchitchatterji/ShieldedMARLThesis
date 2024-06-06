import publicgoodsmany_env
env = publicgoodsmany_env.parallel_env(render_mode="human", n_agents=4, rand_f_between=[0.1,4], max_cycles=10, initial_endowment=10, observe_f=True)

observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(rewards)

env.close()