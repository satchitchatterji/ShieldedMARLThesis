import torch
import numpy as np

class WaterworldSensorWrapper:
    """
    Sensor wrapper for the Waterworld environment. 
    This is to convert the sensor values to a (probabalistic) form that the shield can use.
    """
    def __init__(self, env, output_type="invert"):
        self.env = env
        self.num_inputs = env.observation_spaces[env.agents[0]].shape[0]
        self.output_type = output_type
        if output_type == "invert":
            self.num_sensors = self.num_inputs
            self.translation_func = self.invert_sensor_vals
        elif output_type == "reduce_to_8":
            self.num_sensors = 8*5+2
            self.translation_func = self.reduce_to_8
        else:
            raise NotImplementedError("Only 'invert' and 'reduce_to_8' is supported for now.")            
    
def reduce_to_8(x, use_cuda=True):
    assert len(x) == 162, f"Expected 162 sensor values, got {len(x)}"
    # TODO: use torch instead of numpy to reduce overhead
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    x_new = torch.zeros(8*5+2, device=device) # 5 types of sensors, 8 ranges, 2 for the last two sensors
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
    ranged_x_diff = torch.stack(ranged_x_diff).to(device)
    ranged_x_diff = ranged_x_diff.mean(axis=1)
    x_new[:8*5] = ranged_x_diff.flatten()
    return x_new

    def invert_sensor_vals(self, x):
        x[:len(x)-2] = 1-x[:len(x)-2]
        return x
    
    def __call__(self, x):
        return self.translation_func(x)