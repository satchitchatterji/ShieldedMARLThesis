import torch
import numpy as np
import sys
sys.path.append("../grid_envs")
import parallel_stag_hunt as psh

def get_wrapper(env):
    wrappers = {
        "waterworld": WaterworldSensorWrapper,
        "simple_stag_v0": IdentitySensorWrapper,
        "simple_pd_v0": IdentitySensorWrapper,
        "simple_chicken_v0": IdentitySensorWrapper,
        "markov_stag_hunt": MarkovStagHuntSensorWrapper,
        "centipede": IdentitySensorWrapper,
        "publicgoods": PublicGoodsSensorWrapper,
        "publicgoodsmany": PublicGoodsSensorWrapper
    }
    if env in wrappers.keys():
        return wrappers[env]
    else:
        raise NotImplementedError(f"Sensor wrapper for {env} not implemented.")

class IdentitySensorWrapper:
    """
    Sensor wrapper for the PettingZoo environments. 
    """
    def __init__(self, env, num_sensors=None):
        self.env = env
        if num_sensors is not None:
            self.num_sensors = num_sensors
        else:
            self.num_sensors = env.observation_spaces[env.agents[0]].shape[0]
    
    def __call__(self, x):
        return x
    
class PublicGoodsSensorWrapper:
    """
    Sensor wrapper for the Public Goods Game environment. 
    """
    def __init__(self, env, num_sensors=None):
        self.env = env
        self.num_sensors = num_sensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if env.observe_f:
            self.translation_func = self.obs_with_f
        else:
            self.translation_func = self.obs_without_f
            
        self._reset_values()

    def _reset_values(self):
        self.values_low = None
        self.values_high = None

    def obs_without_f(self, obs):
        return obs
    
    def obs_with_f(self, obs):
        if self.env.num_moves == 0:
            self._reset_values()
        obs = obs.cpu().numpy()
        agent_other = obs[:-1]
        mult = obs[-1]
        if self.values_low is None:
            self.values_low = mult
            self.values_high = mult
        else:
            
            self.values_low = min(mult, self.values_low)
            self.values_high = max(mult, self.values_high)
        
        if self.values_high == self.values_low:
            mult = 0.5
        else:
            mult = (mult - self.values_low) / (self.values_high - self.values_low)
        
        return torch.tensor(np.append(agent_other, mult).astype(np.float32), dtype=torch.float32, device=self.device)
    
    def __call__(self, x):
        # TODO: batch processing
        if len(x.shape) == 1:
            return self.translation_func(x)
        else:
            return torch.stack([self.translation_func(x[i]) for i in range(x.shape[0])])
class MarkovStagHuntSensorWrapper:

    def __init__(self, env, num_sensors=None):
        self.env = env
        self.num_sensors = 6 # based on stag relative position
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translation_func = self.stag_surrounded

    def one_hot_to_obs(self, one_hot_obs):
        GRID_SIZE = psh.GRID_SIZE
        one_hot_obs = one_hot_obs.reshape((GRID_SIZE[0], GRID_SIZE[1], psh.N_OBS_TYPES))
        obs = np.zeros((GRID_SIZE[0], GRID_SIZE[1]), dtype=int)
        for x in range(GRID_SIZE[0]):
            for y in range(GRID_SIZE[1]):
                obs[x][y] = np.argmax(one_hot_obs[x][y])
        return obs
    
    def stag_surrounded(self, obs):
        """
        Returns the relative position of the stag to the agent,
                    if the agent is close to the stag, and
                    if the stag is surrounded by both agents.
        """
        obs = self.one_hot_to_obs(obs.cpu().numpy())
        stag_x, stag_y = np.where(obs == psh.OBS_STAG) # assuming only one stag
        stag_x, stag_y = stag_x[0], stag_y[0]

        agent_self = np.where(obs == psh.OBS_AGENT_SELF)
        agent_other = np.where(obs == psh.OBS_AGENT_OTHER)
        both_agents = np.where(obs == psh.OBS_AGENT_BOTH)
        try:
            agent_self = agent_self if len(agent_self[0]) > 0 else both_agents
            agent_self_x, agent_self_y = agent_self[0][0], agent_self[1][0]

            agent_other = agent_other if len(agent_other[0]) > 0 else both_agents
            agent_other_x, agent_other_y = agent_other[0][0], agent_other[1][0]
        except IndexError as e:
            print(obs)
            print(agent_self, agent_other, both_agents)
            raise e
        # print("\n", stag_x, stag_y, agent_self_x, agent_self_y, agent_other_x, agent_other_y, both_agents, "\n")
        assert agent_self_x is not None, f"Agent self not found in observation: {obs}"

        rel_x = stag_x - agent_self_x
        rel_y = stag_y - agent_self_y
        
        stag_above = rel_x < 0
        stag_below = rel_x > 0
        stag_left = rel_y < 0
        stag_right = rel_y > 0

        # check which agents are in the stag's Moore's neighborhood
        stag_near_self = 0
        if agent_self_x-1 <= stag_x <= agent_self_x+1 and agent_self_y-1 <= stag_y <= agent_self_y+1:
            stag_near_self = 1

        stag_near_other = 0
        if agent_other_x-1 <= stag_x <= agent_other_x+1 and agent_other_y-1 <= stag_y <= agent_other_y+1:
            stag_near_other = 1

        return torch.tensor(np.array([stag_left, stag_right, stag_above, stag_below, stag_near_self, stag_near_other], dtype=np.int32), 
                            dtype=torch.int32, device=self.device)

    def __call__(self, x):
        # TODO: batch processing
        if len(x.shape) == 1:
            return self.translation_func(x)
        else:
            return torch.stack([self.translation_func(x[i]) for i in range(x.shape[0])])
        
class WaterworldSensorWrapper:
    """
    Sensor wrapper for the Waterworld environment. 
    This is to convert the sensor values to a (probabalistic) form that the shield can use.
    """
    def __init__(self, env, output_type="invert"):
        self.env = env
        self.num_inputs = env.observation_spaces[env.agents[0]].shape[0]
        self.output_type = output_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if output_type == "invert":
            self.num_sensors = self.num_inputs
            self.translation_func = self.invert_sensor_vals
        elif output_type == "reduce_to_8":
            self.num_sensors = 8*5+2
            self.translation_func = self.reduce_to_8
        else:
            raise NotImplementedError("Only 'invert' and 'reduce_to_8' is supported for now.")            
    
    def reduce_to_8(self, x, use_cuda=True):
        assert len(x) == 162, f"Expected 162 sensor values, got {len(x)}"
        x_new = torch.zeros(8*5+2, device=self.device) # 5 types of sensors, 8 ranges, 2 for the last two sensors
        x_new[-1] = 1-x[-1]
        x_new[-2] = 1-x[-2]
        x_diff = 1-x[:len(x)-2]
        x_diff = x_diff.reshape((32, -1)) # 32 x 5
        ranges = [[-2,2], [2,6], [6,10], [10,14], [14,18], [18,22], [22,26], [26,30], [30,31]] # 8 ranges
        ranged_x_diff = []
        first_range = torch.stack(list(x_diff[-2:0])+list(x_diff[0:3])+list(x_diff[30:]), axis=-1) 
        ranged_x_diff.append(first_range)
        for r in ranges[1:-1]:
            ranged_x_diff.append(x_diff[r[0]:r[1]+1])
        ranged_x_diff = torch.stack(ranged_x_diff).to(self.device)
        ranged_x_diff = ranged_x_diff.mean(axis=1)
        x_new[:8*5] = ranged_x_diff.flatten()
        return x_new

    def invert_sensor_vals(self, x):
        x[:len(x)-2] = 1-x[:len(x)-2]
        return x
    
    def __call__(self, x):
        # TODO: batch processing
        if len(x.shape) == 1:
            return self.translation_func(x)
        else:
            return torch.stack([self.translation_func(x[i]) for i in range(x.shape[0])])