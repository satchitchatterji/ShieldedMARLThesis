# amended from https://github.com/arjun-prakash/pz_dilemma/blob/main/src/environments/centipede/centipede_v1.py

from gym import spaces
import gym
import gym_safety
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

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
    """Two-player environment for classic centipede game.
    The observation is simply the last opponent action.

    https://www.semanticscholar.org/paper/The-Dynamic-of-Bicycle-Finals%3A-A-Theoretical-and-of-Dilger-Geyer/28ed6c168374bf1866fcdc0f01fa094448a1f009
    https://www.researchgate.net/publication/283119813_Strategic_Behavior_in_Road_Cycling_Competitions
    https://www.mdpi.com/2073-4336/11/3/35

    https://www.econstor.eu/bitstream/10419/167945/1/834230089.pdf
    """

    metadata = {'render.modes': ['human'], "name": "gym_safety"}

    def __init__(self, env_name, n_agents=1, max_cycles=50, render_mode=None, **env_kwargs):
        parallel_env.metadata["name"] = env_name
        
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

    def step(self, actions):

        action = actions[self.agents[0]]
        next_obs, reward, done, info = self._env.step(action)
        
        self.observations = {agent: next_obs for agent in self.agents}
        self.rewards = {agent: reward for agent in self.agents}
        self.terminations = {agent: done for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: info for agent in self.agents}
        self.num_moves += 1

        env_truncated = done or self.num_moves >= self.max_cycles
        if env_truncated:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos
