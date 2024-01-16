import numpy as np

class RandomAgent():
    def __init__(self, num_states, num_actions, num_agents):
        self.observation_type = 'discrete'
        self.action_type = 'discrete'
    
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_agents = num_agents
    
        self.eval_mode = False
        self.rewards = []

    def act(self, states):
        actions = []
        for i in range(self.num_agents):
            actions.append(np.random.randint(0, self.num_actions))
        return actions

    def update_reward(self, reward):
        self.rewards.append(reward)

    def _learn(self, state, action, reward, next_state, next_action):
        pass

    def begin_episode(self):
        pass

    def get_params(self):
        return "Random"

    def set_params(self, params):   
        pass