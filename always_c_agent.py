import numpy as np

class CAgent():
    def __init__(self, num_states, num_actions):
        self.observation_type = 'discrete'
        self.action_type = 'discrete'
    
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_agents = None
    
        self.eval_mode = False
        self.rewards = []

        self.name = "AlwaysC"

    def update_n_agents(self, n_agents):
        self.num_agents = n_agents

    def act(self, states):
        actions = []
        for i in range(self.num_agents):
            actions.append(0)
        return actions

    def update_reward(self, reward):
        self.rewards.append(reward)

    def _learn(self, state, action, reward, next_state, next_action):
        pass

    def begin_episode(self):
        pass

    def get_params(self):
        return "AlwaysC"

    def set_params(self, params):   
        pass