from .base import BaseMARLAlgo
from .dqn_shielded import DQNShielded

class SIQL(BaseMARLAlgo):
    def __init__(self, env, observation_space, n_discrete_actions, action_wrapper, sh_params, alpha, **kwargs):
        super().__init__(env=env, 
                         observation_space=observation_space, 
                         n_discrete_actions=n_discrete_actions, 
                         action_wrapper=action_wrapper, 
                         sh_params=sh_params,
                         alpha=alpha, #TODO: make this param part of algorithm_params
                         **kwargs
                         )

    def create_agents(self):
        for agent in self.env.agents:
            self.agents[agent] = DQNShielded(self.observation_space, 
                                             self.n_discrete_actions, 
                                             sh_params=self.sh_params,
                                             alpha=self.alpha)