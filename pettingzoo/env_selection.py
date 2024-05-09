import sys

from pettingzoo.mpe import simple_spread_v3
sys.path.append("../grid_envs")
import parallel_stag_hunt

sys.path.append("../pettingzoo_dilemma_envs")
from dilemma_pettingzoo import parallel_env as dilemma_parallel_env

ALL_ENVS = {
    "simple_spread_v3": simple_spread_v3.parallel_env,
    "markov_stag_hunt": parallel_stag_hunt,
    "nfg_stag_hunt": dilemma_parallel_env,
}

ALL_ENVS_ARGS = {
    "simple_spread_v3": {"N": 3, "local_ratio": 0.5, "continuous_actions": False},
    "markov_stag_hunt": {},
    "nfg_stag_hunt": {"game": "stag", "num_actions": 2},
}