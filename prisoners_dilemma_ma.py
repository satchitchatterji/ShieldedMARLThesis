# multi-agent prisonners dilemma environment

import numpy as np

class spec:
    def __init__(self):
        self.max_episode_steps = 10

class action_space:
    def __init__(self, n):
        self.n = n

class PrisonersDilemmaMAEnv:
    def __init__(self, agents, render_mode=False):
        self.agents = agents
        self.n_agents = len(agents)
        self.n_actions = 2
        self.n_states = 2
        self.state = np.full((self.n_agents, self.n_agents), None)
        self.action_space = [0, 1]
        self.observation_space = [0, 1]
        self.spec = spec()
        self.action_space = action_space(self.n_actions)

        self.render_mode = render_mode

    def reset(self):
        self.state = np.full((self.n_agents, self.n_agents), None)
        return (self.state,)
    
    def step(self, actions):
        # actions is 2D array of shape (n_agents, n_agents)
        rewards = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                if actions[i][j] == 0 and actions[j][i] == 0:
                    rewards[i][j] = 8
                    rewards[j][i] = 8
                elif actions[i][j] == 0 and actions[j][i] == 1:
                    rewards[i][j] = 0
                    rewards[j][i] = 10
                elif actions[i][j] == 1 and actions[j][i] == 0:
                    rewards[i][j] = 10
                    rewards[j][i] = 0
                elif actions[i][j] == 1 and actions[j][i] == 1:
                    rewards[i][j] = 4
                    rewards[j][i] = 4

        self.state = actions
        
        if self.render_mode:
            self.render()

        return self.state, rewards, False, False, {}
    
    def render(self):
        print(self.state)

    def close(self):
        pass
