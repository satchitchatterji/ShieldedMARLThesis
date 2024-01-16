import numpy as np
from config import config

def calculate_index(coordinates, dimensions):
    # unique index of a point with coordinates 'coordinates'
    # in a nD box of dimension n, with max size 'dimensions'
    idx = coordinates[0]
    for i in range(1, len(coordinates)):
        idx = idx * dimensions[i] + coordinates[i]
    return idx

def discretize_state(observation):
    if config.env == 'CartPole-v1':
        return discretize_cartpole_state(observation)
    elif config.env == 'MountainCar-v0':
        return discretize_mountaincar_state(observation)
    elif config.env == 'PrisonersDilemmaMA':
         return observation
    
def discretize_mountaincar_state(observation):
            
        position_bins = np.linspace(-1.2, 0.6, config.discretization)
        velocity_bins = np.linspace(-0.07, 0.07, config.discretization)
    
        position = np.digitize(observation[0], position_bins)
        velocity = np.digitize(observation[1], velocity_bins)
    
        return calculate_index((position, velocity), [config.discretization]*config.observation_space)

def discretize_cartpole_state(observation):

    cart_position_bins = np.linspace(-2.4, 2.4, config.discretization)
    cart_velocity_bins = np.linspace(-2, 2, config.discretization)
    pole_angle_bins = np.linspace(-0.4, 0.4, config.discretization)
    pole_velocity_bins = np.linspace(-3.5, 3.5, config.discretization)

    cart_position = np.digitize(observation[0], cart_position_bins)
    cart_velocity = np.digitize(observation[1], cart_velocity_bins)
    pole_angle = np.digitize(observation[2], pole_angle_bins)
    pole_velocity = np.digitize(observation[3], pole_velocity_bins)

    return calculate_index((cart_position, cart_velocity, pole_angle, pole_velocity), [config.discretization]*config.observation_space)
