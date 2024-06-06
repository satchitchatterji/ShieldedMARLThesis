import numpy as np

def compute_current_epsilon(eps_start, eps_decay, timestep):
    """
    Compute epsilon decay schedule using multiplicative decay.
    """
    return eps_start * (eps_decay ** timestep)


def compute_eps_decay(eps_start, eps_end, timestep):
    return (eps_end / eps_start) ** (1 / timestep)

# Example usage:
eps_start = 1.0
eps_end = 0.05
timestep = 20000

eps_decay = compute_eps_decay(eps_start, eps_end, timestep)
print(eps_decay)

eps_decay = 0.9999
print(compute_current_epsilon(eps_start, eps_decay, 25000))