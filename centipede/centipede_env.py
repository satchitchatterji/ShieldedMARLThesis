# amended from https://github.com/arjun-prakash/pz_dilemma/blob/main/src/environments/centipede/centipede_v1.py

from gym import spaces
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

import functools

MOVES = {'DEFECT': 0, 'COOPERATE': 1, 'NONE': 2}

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

    metadata = {'render.modes': ['human'], "name": "centipede", "is_parallelizable": True}

    def __init__(self, n_agents=2, initial_endowment=1, growth_rate=2, max_cycles=50, render_mode=None, randomize_players=False):
        self.agents = ["player_" + str(r) for r in range(n_agents)]
        self.possible_agents = self.agents[:]
        self.randomize_players = randomize_players
        self.max_cycles = max_cycles
        print('Centipede!', self.max_cycles)

        self.state = {agent: 2 for agent in self.agents}
        self.initial_endowment = initial_endowment  # Initial value of endowment
        self.growth_rate = growth_rate  # Growth rate of the pot
        self.total_pot = initial_endowment  # Track the total pot value
        self.render_mode = render_mode

        self.action_spaces = {agent: spaces.Discrete(len(MOVES) - 1) for agent in self.agents}
        self.observation_spaces = {agent: spaces.Discrete(len(MOVES)) for agent in self.agents}

        self.reset()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Discrete(len(MOVES))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(MOVES)-1)
        
    def reset(self):
        self.agents = self.possible_agents[:]
        if self.randomize_players:
            self.agents = np.random.permutation(self.agents).tolist()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.observations = {agent: 2 for agent in self.agents}
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.num_moves = 0
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: MOVES["NONE"] for agent in self.agents}
        self.total_pot = self.initial_endowment

        return self.observations, self.infos

    def render(self):
        if self.render_mode == "human":
            print(f"Player actions: {self.state}")
            
    def observe(self, agent):
        flat_obs = np.zeros(len(MOVES))
        flat_obs[self.observations[agent]] = 1
        return flat_obs

    def close(self):
        pass

    def step(self, actions):
        self.rewards = {agent: 0 for agent in self.agents}
        env_truncated = False
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        self.num_moves += 1
        for aidx, agent in enumerate(self.agents):
            action = actions[agent]
            self.state[agent] = action

            if action == MOVES['COOPERATE']:
                self.total_pot += self.growth_rate

            elif action == MOVES['DEFECT']:
                if agent == 'player_0':
                    self.rewards['player_0'] = (self.total_pot / 2) + 1
                    self.rewards['player_1'] = (self.total_pot / 2) - 1
                else:
                    self.rewards['player_0'] = (self.total_pot / 2) - 1
                    self.rewards['player_1'] = (self.total_pot / 2) + 1
                self.terminations = {agent: True for agent in self.agents}
                env_truncated = True

        if self.num_moves >= self.max_cycles:
            for agent in self.agents:
                if not self.terminations[agent]:
                    self.rewards[agent] = self.total_pot / 2
            self.terminations = {agent: True for agent in self.agents}
            env_truncated = True

        self.observations = {agent: self.state[self.agents[1 - self.agents.index(agent)]] for agent in self.agents}
        self.observations = {agent: self.observe(agent) for agent in self.agents}

        self._accumulate_rewards()
        
        if env_truncated:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def _accumulate_rewards(self):
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]