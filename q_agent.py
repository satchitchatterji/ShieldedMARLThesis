import numpy as np

class QAgent(object):
    def __init__(self, num_states, num_actions):
        self.observation_type = 'discrete'
        self.action_type = 'discrete'
        
        self.q = np.zeros((num_states, num_actions))
        self.q_visit_count = np.zeros((num_states, num_actions))

        self.gamma = 0.9
        self.epsilon = 0.07
        self.alpha = lambda state, action: 1.0 / (1.0 + self.q_visit_count[state, action]) 
        
        self.num_states = num_states
        self.num_actions = num_actions

        self.prev_state = None
        self.prev_action = None
        self.reward = None

        self.eval_mode = False
        

    def act(self, state):
        if np.random.rand() < self.epsilon and not self.eval_mode:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q[state])

        if self.prev_action is not None and not self.eval_mode:
            self._learn(self.prev_state, self.prev_action, self.reward, state, action)
        
        self.prev_action = action
        self.prev_state = state

        return action
        
    def update_reward(self, reward):
        self.reward = reward

    def _learn(self, state, action, reward, next_state, next_action):
        # update value function
        target = reward + self.gamma * np.max(self.q[next_state])
        self.q[state, action] += self.alpha(state, action) * (target - self.q[state, action])        
        self.q_visit_count[state, action] += 1

    def begin_episode(self):
        self.prev_state = None
        self.prev_action = None
        self.reward = None

    def get_params(self):
        return self.q
    
    def set_params(self, params):   
        self.q = params