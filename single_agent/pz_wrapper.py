
from gym import spaces
import gym
import gym_safety
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
import math
import functools

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = parallel_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class parallel_env(ParallelEnv):

    metadata = {'render.modes': ['human'], "name": "gym_safety"}

    def __init__(self, env_name, n_agents=1, max_cycles=50, render_mode=None, **env_kwargs):
        assert n_agents == 1, "This environment is designed for single agent only"
        
        parallel_env.metadata["name"] = env_name
        assert env_name == "CartSafe-v0", f"This wrapper is only designed so far for CartSafe-v0 (not {env_name}) because of env customization."
        
        self._env = gym.make(env_name, **env_kwargs)
        self.agents = ["player_" + str(r) for r in range(n_agents)]
        self.possible_agents = self.agents[:]
        self.max_cycles = max_cycles

        self.state = {agent: 2 for agent in self.agents}
        self.render_mode = render_mode

        self.action_spaces = {agent: self._env.action_space for agent in self.agents}
        self.observation_spaces = {agent: self._env.observation_space for agent in self.agents}

        self.reset()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
        
    def reset(self):
        self.num_moves = 0
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        obs = self._env.reset()
        self._env.state[2] -= math.pi
        obs[2] -= math.pi
        self.observations = {agent: obs for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = self.observations

        return self.observations, self.infos

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def compute_reward(self, agent, action):
        angle = self._env.state[2]
        fifteen_degrees = math.pi / 12
        if -fifteen_degrees < angle < fifteen_degrees:
            return 1
        return -1
    
    def step(self, actions):

        action = actions[self.agents[0]]

        reward = self.compute_reward(self.agents[0], action)        
        next_obs, original_reward, done, info = self._env.step(action)
        
        if done:
            reward = 0

        env_truncated = done
        if reward < -100:
            env_truncated = True

        # reward = original_reward

        self.observations = {agent: next_obs for agent in self.agents}
        self.rewards = {agent: reward for agent in self.agents}
        self.terminations = {agent: done or self.num_moves >= self.max_cycles for agent in self.agents}
        self.truncations = {agent: env_truncated for agent in self.agents}
        self.infos = {agent: info for agent in self.agents}
        self.num_moves += 1

        if env_truncated:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos
