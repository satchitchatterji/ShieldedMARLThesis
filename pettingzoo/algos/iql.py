from .base import BaseMARLAlgo
from .dqn_shielded import DQNShielded

class IQL(BaseMARLAlgo):
    def __init__(self, env, observation_space, n_discrete_actions, action_wrapper, algorithm_params, **kwargs):
        super().__init__(env=env, 
                         observation_space=observation_space, 
                         n_discrete_actions=n_discrete_actions, 
                         action_wrapper=action_wrapper,
                         algorithm_params=algorithm_params,
                         **kwargs
                         )

    def create_agents(self):
        self.agents = {}
        for agent in self.env.agents:
            self.agents[agent] = DQNShielded(self.observation_space, self.n_discrete_actions, **self.algorithm_params)
