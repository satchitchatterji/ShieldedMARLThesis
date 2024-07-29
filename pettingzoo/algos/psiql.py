from .base import BaseMARLAlgo
from .better_dqnpls import DQNShielded

class PSIQL(BaseMARLAlgo):
    def __init__(self, env, observation_space, n_discrete_actions, action_wrapper, algorithm_params, **kwargs):
        super().__init__(env=env, 
                         observation_space=observation_space, 
                         n_discrete_actions=n_discrete_actions, 
                         action_wrapper=action_wrapper,
                         algorithm_params=algorithm_params,
                         **kwargs
                         )

    def create_agents(self):
        self.agents[self.env.agents[0]] = DQNShielded(self.observation_space, self.n_discrete_actions, **self.algorithm_params)
        for a, agent in enumerate(self.env.agents):
            if a != 0:
                self.agents[agent] = DQNShielded(self.observation_space, 
                                                 self.n_discrete_actions,
                                                 func_approx=self.agents[self.env.agents[0]].func_approx, 
                                                 **self.algorithm_params)

