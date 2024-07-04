import numpy as np

def compute_current_epsilon(eps_start, eps_decay, timestep):
    """
    Compute epsilon decay schedule using multiplicative decay.
    """
    return eps_start * (eps_decay ** timestep)

def compute_eps_decay(eps_start, eps_end, timestep):
    return (eps_end / eps_start) ** (1 / timestep)

def compute_eps_min_timestep(eps_start, eps_end, eps_decay):
    """
    Compute the timestep at which the current epsilon is below eps_end.
    """
    return np.ceil(np.log(eps_end / eps_start) / np.log(eps_decay))

if __name__ == "__main__":
    eps_start = 1.0
    eps_end = 0.01
    timestep = 25*400

    eps_decay = compute_eps_decay(eps_start, eps_end, timestep)
    print(eps_decay)

    eps_decay = 0.9995
    print(compute_current_epsilon(eps_start, eps_decay, timestep))