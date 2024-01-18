# Playground for Thesis Ideas

This repository is meant to store code that I use to pracitce various themes while reading for my thesis.

## Common files
- config.py : global configuration variables and helper functions
- discretize.py : used to discretize observations from a continuous-observation environment (such as cartpole) for agents that can only deal with discrete observation spaces.

## Classical RL Practice
- cartpole.py : main code that can be used for any openai gym classical control environments (tested with cartpole and mountain climber).
- default_agent.py : skeleton code for an agent in this environment
- ipe_agent.py : (_not implemented_) realised that IPE is not a useful algorithm for classical control tasks
- print_rewards.py : a variety of print functions for rewards 
- q_agent.py : Q-Learning agent for gym environements
- sarsa_agent.py : SARSA-Learning agent for gym environments

## MARL Practice
- marl_pd.py : main code based on control.py using the PrisonersDilemmaMAEnv, but also includes plotting code. It currently supports optimizing for selfish utility, social welfare and maximum and minimum utility disparity.
- prisoners_dilemma_ma.py : environment with similar API to gym, can host _n_ agents for any normal form game
- always_c_agent.py : classical agent for iterated PD: _Always Cooperate_
- always_d_agent.p : classical agent for iterated PD: _Always Defect_
- pd_q_agent.py : Q-Learning agent for MA NFG environments, compatible with PrisonersDilemmaMAEnv
- pd_sarsa_agent.py : SARSA-Learning agent for MA NFG environments, compatible with PrisonersDilemmaMAEnv
- random_agent.py : test agent for iterated PD: _Random_
- tittat_agent.py : classical agent for iterated PD: _Tit for Tat_