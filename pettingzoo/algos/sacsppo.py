from .base import BaseMARLAlgo
from .ppo_shielded import PPOShielded

class SACSPPO(BaseMARLAlgo):
    def __init__(self, env, observation_space, n_discrete_actions, action_wrapper, algorithm_params, alpha, sh_params, sensor_wrapper, **kwargs):
        super().__init__(env=env, 
                         observation_space=observation_space, 
                         n_discrete_actions=n_discrete_actions, 
                         action_wrapper=action_wrapper,
                         algorithm_params=algorithm_params,
                         alpha=alpha,
                         sh_params=sh_params,
                         sensor_wrapper=sensor_wrapper,
                         **kwargs
                         )

    def create_agents(self):
        self.agents[self.env.agents[0]] = PPOShielded(state_dim=self.observation_space, 
                                        action_dim=self.n_discrete_actions, 
                                        alpha=self.alpha, 
                                        policy_safety_params=self.sh_params,
                                        policy_kw_args={"shield_params":self.sh_params, "get_sensor_value_ground_truth":self.sensor_wrapper},
                                        **self.algorithm_params)
        for a, agent in enumerate(self.env.agents):
            if a != 0:
                self.agents[agent] = PPOShielded(state_dim=self.observation_space, 
                                        action_dim=self.n_discrete_actions, 
                                        alpha=self.alpha,
                                        policy_safety_params=self.sh_params,
                                        policy_kw_args={"shield": self.agents[self.env.agents[0]].policy.shield, "get_sensor_value_ground_truth":self.sensor_wrapper},
                                        **self.algorithm_params)
                self.agents[agent].set_policy(self.agents[self.env.agents[0]].policy)

