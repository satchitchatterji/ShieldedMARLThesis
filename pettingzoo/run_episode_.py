raise NotImplementedError("This file was supposed to be made to facilitate running episodes in AEC or Parallel pettingzoo environments, but it was not needed.")
import run_episode_par
import run_episode_aec
from pettingzoo import AECEnv, ParallelEnv

def run_episode(env, algo, max_cycles, ep=0):
    if issubclass(type(env) , AECEnv):
        return run_episode_aec.run_episode(env, algo, max_cycles, ep)
    elif issubclass(type(env), ParallelEnv):
        return run_episode_par.run_episode(env, algo, max_cycles, ep)
    else:
        raise ValueError("env must be an instance of AECEnv or ParallelEnv")
    
def eval_episode(env, algo, max_cycles, ep=0, safety_calculator=None, save_wandb=True):
    if issubclass(type(env) , AECEnv):
        return run_episode_aec.eval_episode(env, algo, max_cycles, ep, safety_calculator, save_wandb)
    elif issubclass(type(env), ParallelEnv):
        return run_episode_par.eval_episode(env, algo, max_cycles, ep, safety_calculator, save_wandb)
    else:
        raise ValueError("env must be an instance of AECEnv or ParallelEnv")