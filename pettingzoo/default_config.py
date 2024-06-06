from dataclasses import dataclass

@dataclass
class Config:
    algo: str = "IQL"
    env: str = "nfg_stag_hunt"
    max_cycles: int = 25
    max_eps: int = 500
    eval_every: int = 50
    n_eval: int = 5
    shield_alpha: float = 1.0
    shield_file: str = "default"
    shield_version: int = 0
    update_timestep: int = 5
    train_epochs: int = 10
    gamma: float = 0.99
    eps_clip: float = 0.1
    lr_actor: float = 0.001
    lr_critic: float = 0.001
    buffer_size: int = 1000
    batch_size: int = 64
    lr: float = 0.001
    tau: float = 0.01
    target_update_type: str = "hard"
    eps_min: float = 0.05
    eps_decay: float = 0.995

    def update(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Example of creating a config object
default_config = Config()
