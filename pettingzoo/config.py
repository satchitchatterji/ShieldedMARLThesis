from algos import *
from env_selection import ALL_ENVS

import argparse
parser = argparse.ArgumentParser()

# env and algo params

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

# Model Pararms
parser.add_argument("--update_timestep",
                    type=int,
                    default=25,
                    help="Update policy (PPO) / target network (DQN) every n timesteps."
                    )

parser.add_argument("--epochs",                                     # Also used in DQN
                    type=int,
                    default=50,
                    help="Update policy for K epochs in one PPO/DQN update."
                    )


parser.add_argument("--gamma",                                      # Also used in DQN
                    type=float,
                    default=0.99,
                    help="Discount factor."
                    )

# PPO Params
parser.add_argument("--eps_clip",
                    type=float,
                    default=0.2,
                    help="Clip parameter for PPO."
                    )


parser.add_argument("--lr_actor",
                    type=float,
                    default=0.001,
                    help="Learning rate for actor network."
                    )

parser.add_argument("--lr_critic",
                    type=float,
                    default=0.001,
                    help="Learning rate for critic network."
                    )

# DQN Params
parser.add_argument("--buffer_size",
                    type=int,
                    default=10000,
                    help="Size of the replay buffer."
                    )

parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help="Batch size for training."
                    )

parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="Learning rate for the Q network."
                    )

parser.add_argument("--tau",
                    type=float,
                    default=0.001,
                    help="Soft update parameter."
                    )

parser.add_argument("--epsilon_min",
                    type=float,
                    default=0.1,
                    help="Min epsilon for epsilon greedy policy."
                    )

parser.add_argument("--epsilon_decay",
                    type=float,
                    default=0.995,
                    help="Decay rate for epsilon."
                    )


config = parser.parse_args()