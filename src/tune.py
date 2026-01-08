"""
Optuna tuning script for Hashi GNN model.
Runs trials sequentially - allows num_workers > 0 for data loading.
"""
import argparse
import optuna
import yaml
import torch
import mlflow
from typing import Dict, Any

from .utils import get_device
from .engine import Trainer
from .callbacks import MLflowCallback, OptunaPruningCallback
from .tune_space import expand_trial_config
from .data import RandomHashiAugment


def run_trial(
    trial: optuna.Trial,
    base_config: Dict[str, Any],
    search_space: Dict[str, Any],
    device: torch.device,
    experiment_name: str
) -> float:
    """Run a single trial: sample params, build data, train, evaluate."""
    print(f"Trial {trial.number}: Starting trial")
    
    # 1. Sample from search space
    trial_params = {}
    for path, spec in search_space.items():
        if isinstance(spec, dict) and 'choices' in spec:
            trial_params[path] = trial.suggest_categorical(path, spec['choices'])
        elif isinstance(spec, dict) and 'low' in spec:
            if spec.get('type') == 'int':
                trial_params[path] = trial.suggest_int(path, spec['low'], spec['high'], log=spec.get('log', False))
            else:
                trial_params[path] = trial.suggest_float(path, spec['low'], spec['high'], log=spec.get('log', False))
        else:
            trial_params[path] = spec  # Constant
    
    # 2. Expand hierarchical constraints and merge into config
    full_config = {
        'data': base_config['data'].copy(),
        'model': base_config['model'].copy(),
        'training': base_config['training'].copy()
    }
    expanded_params = expand_trial_config(trial_params)
    full_config['model'].update(expanded_params)
    
    # 3. Define Callbacks
    callbacks = [
        MLflowCallback(
            experiment_name=experiment_name,
            run_name=f"trial_{trial.number}",
            params=trial_params,
            nested=True
        ),
        OptunaPruningCallback(trial=trial, monitor="val_acc")
    ]

    # 4. Setup augmentation (enabled by default like in training)
    train_transform = None
    aug_config = full_config['training'].get('augmentation', {})
    if aug_config.get('enabled', True):
        train_transform = RandomHashiAugment(
            stretch_prob=aug_config.get('stretch_prob', 0.5),
            max_stretch=aug_config.get('max_stretch', 3)
        )

    # 5. Initialize Trainer
    trainer = Trainer(full_config, device, callbacks=callbacks)

    # 6. Start Training
    try:
        trainer.train(train_transform=train_transform)
    except optuna.exceptions.TrialPruned:
        # Re-raise pruning exception to be handled by Optuna
        raise
    
    # Return the optimization metric (best validation accuracy reached)
    # Note: OptunaPruningCallback handles reporting during training.
    # Here we return the final best accuracy from the trial for Optuna to use.
    # We'll need to make sure we can access the best accuracy. 
    # For now, let's assume we want to return the last epoch's validation accuracy or 
    # keep track of the best in a way that's accessible.
    
    # Let's find the best val_acc from the trial if possible, or just return the last.
    # Actually, Optuna handles the best value if we return the metric from each epoch 
    # but here we return once at the end.
    
    # A better way might be to have the trainer return the best metrics or 
    # store them in the trainer object.
    
    # Let's check how to get the best accuracy from the trainer.
    # I'll update Trainer to store the best val metrics.
    
    result = getattr(trainer, 'best_val_acc', 0.0)
    print(f"Trial {trial.number}: Completed with best_val_acc = {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Tune Hashi GNN using Optuna.")
    parser.add_argument("--tune_config", type=str, default="configs/tune_config.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.tune_config, 'r') as f:
        tune_config = yaml.safe_load(f)

    # Use the standalone config directly
    base_config = tune_config
    search_space = tune_config['search_space']

    device = get_device(args.device or base_config['training'].get('device', 'auto'))

    # Setup sampler
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True)

    # Setup pruner from config
    pruner_config = tune_config.get('pruner', {})
    pruner_type = pruner_config.get('type', 'median')
    if pruner_type == 'median':
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_config.get('n_startup_trials', 5),
            n_warmup_steps=pruner_config.get('n_warmup_steps', 10)
        )
    else:
        # Default to no pruning if unknown type
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=tune_config['study_name'],
        direction=tune_config['direction'],
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    # Setup MLflow experiment
    print(f"Tune.py: Setting up MLflow experiment '{tune_config['study_name']}'")
    mlflow.set_experiment(tune_config['study_name'])

    print("Tune.py: Starting parent run 'Optuna_Study_Parent'")
    with mlflow.start_run(run_name="Optuna_Study_Parent") as parent_run:
        print(f"Tune.py: Parent run started with ID: {parent_run.info.run_id}")

        # Log study parameters
        mlflow.log_param("n_trials", tune_config['n_trials'])
        mlflow.log_param("direction", tune_config['direction'])
        print("Tune.py: Logged study parameters")

        study.optimize(
            lambda trial: run_trial(
                trial, base_config, search_space, device,
                tune_config['study_name']
            ),
            n_trials=tune_config['n_trials'],
            n_jobs=1  # Sequential trials - allows num_workers > 0
        )

        # Log best results to parent run
        mlflow.log_metric("best_value", study.best_trial.value)
        for key, value in study.best_trial.params.items():
            mlflow.log_param(f"best_{key}", value)
        print("Tune.py: Logged best results to parent run")

    print(f"\nBest trial: {study.best_trial.value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
