from .base import BaseMARLAlgo
from .ppo_shielded import PPOShielded

class ACSPPO(BaseMARLAlgo):
    def __init__(self, env, observation_space, n_discrete_actions, action_wrapper, algorithm_params, **kwargs):
        super().__init__(env=env, 
                         observation_space=observation_space, 
                         n_discrete_actions=n_discrete_actions, 
                         action_wrapper=action_wrapper,
                         algorithm_params=algorithm_params,
                         **kwargs
                         )
        self.sensor_wrapper = lambda x: x

    def create_agents(self):
        for agent in self.env.agents:
            self.agents[agent] = PPOShielded(state_dim=self.observation_space, 
                                        action_dim=self.n_discrete_actions, 
                                        policy_kw_args={"get_sensor_value_ground_truth":self.sensor_wrapper},
                                        **self.algorithm_params)
        for a, agent in enumerate(self.env.agents):
            if a != 0:
                self.agents[agent].set_policy(self.agents[self.env.agents[0]].policy)
