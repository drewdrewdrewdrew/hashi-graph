import optuna
from typing import Dict, Any

def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define your search space here using trial.suggest_...
    Values not specified here will be lifted from the base experiment YAML.
    """
    params = {}

    # Loss Weights
    params['loss_weights'] = {
        'ce': trial.suggest_float("ce_weight", 0.5, 2.0),
        'degree': trial.suggest_float("deg_weight", 0.0, 1.0),
        'crossing': trial.suggest_float("cross_weight", 0.0, 1.0),
        'verify': trial.suggest_float("ver_weight", 0.0, 0.5)
    }

    # Training Dynamics
    params['learning_rate'] = trial.suggest_categorical("learning_rate", [0.001, 0.005, 0.0005])
    
    params['training'] = {
        'epochs': 15,           # Fixed value
        'warmup_epochs': 15,    # Fixed value
        'cooldown_epochs': 0    # Fixed value
    }

    # Data size
    params['data'] = {
        'limit': 1000           # Fixed value
    }

    return params



