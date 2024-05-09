from algos import *
from env_selection import ALL_ENVS

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--algo",
                    type=str,
                    default="IQL",
                    help=f"Training style to use. Options are {ALL_ALGORITHMS.keys()}"
                    )

parser.add_argument("--env",
                    type=str,
                    default="nfg_stag_hunt",
                    help=f"Environment to use. Options are {ALL_ENVS.keys()}"
                    )

parser.add_argument("--max_cycles",
                    type=int,
                    default=100,
                    help="Number of cycles to run the environment for."
                    )

parser.add_argument("--max_eps",
                    type=int,
                    default=100,
                    help="Number of episodes to train for."
                    )

parser.add_argument("--eval_every",
                    type=int,
                    default=10,
                    help="Number of episodes between evaluations."
                    )

parser.add_argument("--n_eval",
                    type=int,
                    default=5,
                    help="Number of episodes to evaluate for each evaluation."
                    )

parser.add_argument("--shield_alpha",
                    type=float,
                    default=1.0,
                    help="Alpha value for shield."
                    )

config = parser.parse_args()