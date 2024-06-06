import wandb
from default_config import default_config

sweep_config = {
    'program': 'train.py',
    'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'eval_mean_reward',
        'goal': 'maximize'
    },
    'parameters': {
        'algo': {
            'values': ['ACSPPO']
        },
        'update_time_step': {
            'values': [100, 500, 1000]
        },
        'train_epochs': {
            'values': [10, 20]
        },
        'gamma': {
            'values': [0.99, 0.9]
        },
        'eps_clip': {
            'values': [0.1, 0.2]
        },
        'lr_actor': {
            'values': [0.001, 0.0001]
        },
        'lr_critic': {
            'values': [0.001, 0.0001]
        },
    }
}
# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project=f'ppo_sweep_{default_config.env}')
print(f"Sweep ID: {sweep_id}")
