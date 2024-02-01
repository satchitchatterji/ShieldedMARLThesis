# multi-agent prisonners dilemma environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class spec:
    def __init__(self):
        self.max_episode_steps = 100

class action_space:
    def __init__(self, n):
        self.n = n

class PrisonersDilemmaMAEnv(gym.Env):
    raise NotImplementedError
    metadata = {'render.modes': ['human']}
    def __init__(self, agents, render_mode=False):
        super(PrisonersDilemmaMAEnv, self).__init__()
        self.agents = agents
        self.n_agents = len(agents)
        self.n_actions = 2
        self.n_states = 2
        self.state = np.full((self.n_agents, self.n_agents), None)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiDiscrete(np.full((self.n_agents, self.agents), 2))
        self.spec = spec()
        # self.action_space = action_space(self.n_actions)

        self.render_mode = render_mode

        # utilities is a 2D array of shape (4, 2)
        # NFG format: {TL, TR, BL, BR}
        self.utilities = [[8,8], [0,10], [10,0], [4,4]] # prisoner's dilemma, [cooperate, defect]
        # self.utilities = [[4,4], [0,3], [3,0], [2,2]] # stag hunt, [stag, hare]
        # self.utilities = [[0,0], [7,2], [2,7], [6,6]] # chicken, [straight, swerve]
        
    def reset(self, seed=0):
        self.state = np.full((self.n_agents, self.n_agents), None)
        return (self.state, {})
    
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
        
        if self.render_mode:
            self.render()

        return self.state, rewards, False, False, {}
    
    def render(self):
        print(self.state)

    def close(self):
        pass


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env
    from pd_q_agent import PDQAgent

    env = PrisonersDilemmaMAEnv([PDQAgent(2,2),PDQAgent(2,2)])
    
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
    # env = PrisonersDilemmaMAEnv()
    # env.reset()
    # env.step()
    # env.render()
    # env.close()