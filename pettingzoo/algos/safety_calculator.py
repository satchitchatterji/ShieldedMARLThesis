from pls.shields.shields import Shield
from .base import BaseMARLAlgo
import torch

class SafetyCalculator:
    def __init__(self, sh_params):
        self.shield = Shield(**sh_params)
    
    def compute_safety(self, state, model):
        state = torch.FloatTensor(state).to(model.device)
        action_probs = model.get_action_probs(state)
        sensor_value = self.shield.get_sensor_values(state)
        policy_safety = self.shield.get_policy_safety(sensor_value.unsqueeze(0), action_probs.unsqueeze(0)).squeeze(0)
        return policy_safety
    
    def compute_safety_marl(self, states, marl_model):
        safeties = {agent: self.compute_safety(states[agent], model) for agent, model in marl_model.agents.items()}
        return safeties

    def __call__(self, state, model):
        if issubclass(type(model), BaseMARLAlgo):
            return self.compute_safety_marl(state, model)
        else:
            return self.compute_safety(state, model)