from typing import Any


class WaterworldSensorWrapper:
    
    def __init__(self, env, output_type="invert"):
        self.env = env
        self.num_inputs = env.observation_spaces[env.agents[0]].shape[0]
        self.output_type = output_type
        if output_type == "invert":
            self.num_sensors = self.num_inputs
            self.translation_func = self.invert_sensor_vals
        else:
            raise NotImplementedError("Only invert is supported for now.")            
    
    def invert_sensor_vals(self, x):
        x[:len(x)-2] = 1-x[:len(x)-2]
        return x
    
    def __call__(self, x):
        return self.translation_func(x)