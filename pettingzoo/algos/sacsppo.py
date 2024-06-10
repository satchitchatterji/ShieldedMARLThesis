from .base import BaseMARLAlgo
from .ppo_shielded import PPOShielded
import numpy as np

class SACSPPO(BaseMARLAlgo):
    def __init__(self, env, observation_space, n_discrete_actions, action_wrapper, algorithm_params, alpha, sh_params, sensor_wrapper, shielded_ratio=1.0, **kwargs):
        super().__init__(env=env, 
                         observation_space=observation_space, 
                         n_discrete_actions=n_discrete_actions, 
                         action_wrapper=action_wrapper,
                         algorithm_params=algorithm_params,
                         alpha=alpha,
                         sh_params=sh_params,
                         sensor_wrapper=sensor_wrapper,
                        shielded_ratio=shielded_ratio,
                         **kwargs
                         )

    def create_agents(self):
        for a, agent in enumerate(self.env.agents):
            self.agents[agent] = PPOShielded(state_dim=self.observation_space, 
                                    action_dim=self.n_discrete_actions, 
                                    alpha=self.alpha,
                                    policy_safety_params=self.sh_params,
                                    policy_kw_args={"shield_params":self.sh_params, "get_sensor_value_ground_truth":self.sensor_wrapper},
                                    **self.algorithm_params)
            self.agents[agent].set_policy_critic(self.agents[self.env.agents[0]].policy.critic)
            self.agents[agent].set_policy_actor(self.agents[self.env.agents[0]].policy.actor)

        if self.shielded_ratio == 1.0:
            return
        
        critic = self.agents[self.env.agents[0]].policy.critic
        actor = self.agents[self.env.agents[0]].policy.actor

        n_unshielded = np.round((1-self.shielded_ratio) * len(self.env.agents))
        print(f"\nINFO: Creating {n_unshielded} unshielded agents out of {len(self.env.agents)} total agents.")
        print(f"\t> Shielded ratio (requested): {n_unshielded/len(self.env.agents)} ({1-self.shielded_ratio})\n")
        
        unshielded_agents = np.random.choice(self.env.agents, int(n_unshielded), replace=False)
        for agent in unshielded_agents:
            self.agents[agent] = PPOShielded(state_dim=self.observation_space, 
                                            action_dim=self.n_discrete_actions, 
                                            policy_kw_args={"get_sensor_value_ground_truth":self.sensor_wrapper},
                                            policy_safety_params={},
                                            **self.algorithm_params)

            self.agents[agent].set_policy_critic(critic)
            self.agents[agent].set_policy_actor(actor)