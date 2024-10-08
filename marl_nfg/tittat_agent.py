import numpy as np

class TitForTatAgent(object):
    def __init__(self, num_states, num_actions):
        self.observation_type = 'discrete'
        self.action_type = 'discrete'
        self.learning = False

        self.num_states = num_states
        self.num_actions = num_actions
        self.num_agents = None

        self.prev_states = None
        self.eval_mode = False

        self.rewards = []
        
        self.name = "Tit4Tat"

    def update_n_agents(self, n_agents):
        self.num_agents = n_agents
        self.prev_states = [None]*self.num_agents

    def act(self, other_states):
        if self.num_agents is None:
            raise Exception("Number of agents not set. Call update_n_agents() first.")

        actions = []

        for i in range(self.num_agents):
            if self.prev_states[i] is None:
                actions.append(0)
            else:
                actions.append(self.prev_states[i])
        self.prev_states = other_states
        # print(other_states, actions)
        return actions
                    
    def update_reward(self, reward, *args, **kwargs):
        self.rewards.append(reward)

    def _learn(self, state, action, reward, next_state, next_action):
        pass

    def begin_episode(self):
        self.prev_states = [None]*self.num_agents
        pass

    def get_params(self):
        return "Tit4Tat"

    def set_params(self, params):   
        pass