# multi-agent prisonners dilemma environment

import numpy as np

class spec:
    def __init__(self):
        self.max_episode_steps = 100

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

        self.state_history = []
        self.rewards_history = []

        self.clock = 0

        # utilities is a 2D array of shape (4, 2)
        # NFG format: {TL, TR, BL, BR}
        # self.utilities = [[8,8], [0,10], [10,0], [4,4]] # prisoner's dilemma, [cooperate, defect]
        # self.utilities = [[4,4], [0,3], [3,0], [2,2]] # stag hunt, [stag, hare]
        # self.utilities = [[0,0], [7,2], [2,7], [6,6]] # chicken, [straight, swerve]
        self.utilities = [[1,1], [-1,-1], [-1,-1], [1,1]] # coordination, [A, B]
        
    def reset(self):
        self.clock = 0
        self.state = np.full((self.n_agents, self.n_agents), None)
        return (self.state,)
    
    def step(self, actions):
        # actions is 2D array of shape (n_agents, n_agents)
        # cooperate means tell the truth (0)
        # defect means lie (1)
        # Nash equilibrium is to defect
        # if both cooperate, both get 8
        # if both defect, both get 4
        # if one cooperates and the other defects, the defector gets 10 and the cooperator gets 0
        rewards = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                if actions[i][j] == 0 and actions[j][i] == 0:
                    rewards[i][j] = self.utilities[0][0]
                    rewards[j][i] = self.utilities[0][1]
                elif actions[i][j] == 0 and actions[j][i] == 1:
                    rewards[i][j] = self.utilities[1][0]
                    rewards[j][i] = self.utilities[1][1]
                elif actions[i][j] == 1 and actions[j][i] == 0:
                    rewards[i][j] = self.utilities[2][0]
                    rewards[j][i] = self.utilities[2][1]
                elif actions[i][j] == 1 and actions[j][i] == 1:
                    rewards[i][j] = self.utilities[3][0]
                    rewards[j][i] = self.utilities[3][1]

        self.state = actions
        self.state_history.append(self.state.copy())
        self.rewards_history.append(rewards.copy())

        if self.render_mode:
            self.render()

        self.clock += 1
        terminal = self.clock >= self.spec.max_episode_steps

        return self.state, rewards, terminal, False, {}
    
    def render(self):
        print(self.clock)
        print(self.state)

    def close(self):
        pass
