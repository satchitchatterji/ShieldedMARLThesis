from .base import BaseMARLAlgo
from .ppo_shielded import PPOShielded

class SIPPO(BaseMARLAlgo):
    def __init__(self, env, observation_space, n_discrete_actions, action_wrapper, algorithm_params, sh_params, alpha, **kwargs):
        super().__init__(env=env, 
                         observation_space=observation_space, 
                         n_discrete_actions=n_discrete_actions, 
                         action_wrapper=action_wrapper,
                         algorithm_params=algorithm_params,
                         sh_params=sh_params,
                         alpha=alpha,
                         **kwargs
                         )
        self.sensor_wrapper = lambda x: x
        
    def create_agents(self):
        for agent in self.env.agents:
            self.agents[agent] = PPOShielded(state_dim=self.observation_space, 
                                            action_dim=self.n_discrete_actions, 
                                            alpha=self.alpha,
                                            policy_safety_params=self.sh_params,
                                            policy_kw_args={"shield_params":self.sh_params, "get_sensor_value_ground_truth":self.sensor_wrapper},
                                            **self.algorithm_params)