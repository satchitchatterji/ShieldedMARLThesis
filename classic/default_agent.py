import numpy as np

class Agent(object):
    def __init__(self):
        self.w = np.random.rand(4) * 2 - 1

    def act(self, observation):
        return 1 if np.dot(self.w, observation) > 0 else 0

    def learn(self, state, action, reward, next_state):
        pass

    def update_policy(self, new_w):
        pass

    def get_params(self):
        return self.w