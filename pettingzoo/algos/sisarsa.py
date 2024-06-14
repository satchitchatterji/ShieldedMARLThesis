from .base import BaseMARLAlgo
from .sarsa_shielded import SARSAShielded
import numpy as np

class SISARSA(BaseMARLAlgo):
    def __init__(self, env, observation_space, n_discrete_actions, action_wrapper, sh_params, alpha, algorithm_params,  shielded_ratio=1.0, **kwargs):
        super().__init__(env=env, 
                         observation_space=observation_space, 
                         n_discrete_actions=n_discrete_actions, 
                         action_wrapper=action_wrapper, 
                         sh_params=sh_params,
                         algorithm_params=algorithm_params,
                         alpha=alpha, #TODO: make this param part of algorithm_params
                         shielded_ratio=shielded_ratio,
                         **kwargs
                         )

    def create_agents(self):
        for agent in self.env.agents:
            self.agents[agent] = SARSAShielded(self.observation_space, 
                                             self.n_discrete_actions, 
                                             shield_params=self.sh_params,
                                             alpha=self.alpha,
                                             **self.algorithm_params
                                             )
        
        if self.shielded_ratio == 1.0:
            return
        
        n_unshielded = np.round((1-self.shielded_ratio) * len(self.env.agents))
        print(f"\nINFO: Creating {n_unshielded} unshielded agents out of {len(self.env.agents)} total agents.")
        print(f"\t> Shielded ratio (requested): {n_unshielded/len(self.env.agents)} ({1-self.shielded_ratio})\n")

        unshielded_agents = np.random.choice(self.env.agents, int(n_unshielded), replace=False)
        for agent in unshielded_agents:
            self.agents[agent] = SARSAShielded(self.observation_space, 
                                             self.n_discrete_actions,
                                             shield=None,
                                             shield_params=None,
                                             **self.algorithm_params)

        print([self.agents[agent].shield for agent in self.agents])