"""
Optuna tuning script for Hashi GNN model.
Uses a verbose YAML search space with hierarchical constraints.
"""
import argparse
import optuna
import yaml
import torch
import multiprocessing as mp
import mlflow
from pathlib import Path
from typing import Dict, Any

from .utils import load_config, get_device
from .engine import TrainingEngine, get_masking_rate
from .tracking import MLflowTracker
from .tune_space import expand_trial_config

def run_trial_in_subprocess(config: Dict[str, Any], trial_params: Dict[str, Any], trial_num: int, device_str: str, mlflow_info: Dict[str, str]) -> Any:
    """Run a single trial in a subprocess for clean memory."""
    device = torch.device(device_str)
    
    # Expand hierarchical constraints into the model config
    full_config = config.copy()
    full_config['model'].update(expand_trial_config(trial_params))
    
    # Setup Engine and Tracker
    engine = TrainingEngine(full_config, device)
    tracker = MLflowTracker(mode="tune", experiment_name=mlflow_info['experiment_name'])
    
    with tracker.start_trial_run(trial_num=trial_num, params=trial_params, parent_run_id=mlflow_info.get('parent_run_id')):
        # Setup Data
        train_loader = engine.create_dataloader(split='train', use_cache=True)
        
        # Initialize Model
        model = engine.create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=full_config['training']['learning_rate'])
        
        # Training Loop
        epochs = full_config['training']['epochs']
        best_acc = 0.0
        epoch_results = []
        
        for epoch in range(1, epochs + 1):
            m_rate = get_masking_rate(epoch, full_config['training']['masking'], epochs)
            
            metrics = engine.run_epoch(
                model, train_loader, training=True, 
                optimizer=optimizer, masking_rate=m_rate
            )
            
            tracker.log_epoch(metrics, step=epoch)
            epoch_results.append((epoch, metrics.accuracy))
            best_acc = max(best_acc, metrics.accuracy)
            
    return best_acc, epoch_results

def objective(trial: optuna.Trial, base_config: Dict[str, Any], tune_config: Dict[str, Any], device: torch.device, parent_run_id: str) -> float:
    # 1. Sample from search space
    trial_params = {}
    ss = tune_config.get('search_space', {})
    for path, spec in ss.items():
        if isinstance(spec, dict) and 'choices' in spec:
            trial_params[path] = trial.suggest_categorical(path, spec['choices'])
        elif isinstance(spec, dict) and 'low' in spec:
            if spec.get('type') == 'int':
                trial_params[path] = trial.suggest_int(path, spec['low'], spec['high'], log=spec.get('log', False))
            else:
                trial_params[path] = trial.suggest_float(path, spec['low'], spec['high'], log=spec.get('log', False))
        else:
            trial_params[path] = spec # Constant
            
    device_str = str(device)
    mlflow_info = {
        'experiment_name': tune_config['study_name'],
        'parent_run_id': parent_run_id
    }
    
    # 2. Run in subprocess
    ctx = mp.get_context('spawn')
    with ctx.Pool(1) as pool:
        result = pool.apply(run_trial_in_subprocess, (base_config, trial_params, trial.number, device_str, mlflow_info))
    
    best_acc, epoch_results = result
    
    # 3. Report back to Optuna for pruning
    for epoch, acc in epoch_results:
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_acc

def main():
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Tune Hashi GNN using Optuna.")
    parser.add_argument("--tune_config", type=str, default="configs/tune_config.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.tune_config, 'r') as f:
        tune_config = yaml.safe_load(f)
    base_config = load_config(tune_config['base_config_path'])
    
    device = get_device(args.device or base_config['training'].get('device', 'auto'))
    tracker = MLflowTracker(mode="tune", experiment_name=tune_config['study_name'])
    
    sampler = optuna.samplers.TPESampler(multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=tune_config['study_name'],
        direction=tune_config['direction'],
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    with tracker.start_parent_run(run_name="Optuna_Study_Parent") as parent_run:
        parent_run_id = parent_run.info.run_id
        study.optimize(
            lambda trial: objective(trial, base_config, tune_config, device, parent_run_id),
            n_trials=tune_config['n_trials']
        )

    print(f"\nBest trial: {study.best_trial.value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
