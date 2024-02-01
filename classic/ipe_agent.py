import numpy as np

class IPEAgent(object):
    def __init__(self, num_states, num_actions):
        raise NotImplementedError
        self.observation_type = 'discrete'
        
        self.value = np.zeros(num_states)
        self.policy = np.zeros((num_states, num_actions))
        self.policy.fill(1.0 / num_actions)
        self.gamma = 0.9

        self.num_states = num_states
        self.num_actions = num_actions

    def act(self, observation):
        return np.argmax(self.policy[observation])
    
    def learn(self, state, action, reward, next_state):
        # update value function
        for s in range(self.num_states):
            self.value[s] = (reward + self.gamma * self.value[next_state] - self.value[state])


        # self.value[state] += 0.1 * (reward + 0.9 * self.value[next_state] - self.value[state])
        

    def get_params(self):
        return self.value, self.policy