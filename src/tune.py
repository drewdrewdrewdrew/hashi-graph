"""
Optuna tuning script for Hashi GNN model.
Uses a verbose YAML search space with dot-notation for deep-lifting.
Each trial runs in a subprocess to ensure clean MPS memory state.
"""
import argparse
import optuna
import yaml
import mlflow
import torch
import multiprocessing as mp
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Any

# Import utilities from the main training script
from .train import (
    load_config, get_device, run_epoch, 
    custom_collate_with_conflicts, get_masking_rate
)
from .data import HashiDataset, RandomHashiAugment
from .models import TransformerEdgeClassifier


def get_suggestion(trial, name, spec):
    """Parses a verbose specification from YAML and calls the correct Optuna method."""
    if not isinstance(spec, dict):
        return spec # Constant value
    
    stype = spec.get('type').lower().replace('_', '')
    
    # Float / Uniform / Log-Uniform
    if stype in ['float', 'uniform', 'loguniform']:
        low = float(spec['low'])
        high = float(spec['high'])
        return trial.suggest_float(name, low, high, log=(stype == 'loguniform' or spec.get('log', False)))
    
    # Integer
    if stype == 'int':
        low = int(spec['low'])
        high = int(spec['high'])
        return trial.suggest_int(name, low, high, log=spec.get('log', False))
    
    # Categorical / Choice
    if stype in ['categorical', 'choice']:
        return trial.suggest_categorical(name, spec['choices'])
    
    return spec.get('value', spec)


def set_nested_value(config, key_path, value):
    """Sets a value in a nested dict using a dot-notated path (e.g. 'training.loss_weights.ce')"""
    keys = key_path.split('.')
    curr = config
    for key in keys[:-1]:
        if key not in curr:
            curr[key] = {}
        curr = curr[key]
    curr[keys[-1]] = value


def run_trial_in_subprocess(config: Dict, trial_params: Dict, device_str: str, mlflow_info: Dict[str, str]) -> Any:
    """
    Run a single trial. Called in a subprocess for clean MPS memory.
    Returns the best accuracy achieved.
    """
    from contextlib import nullcontext
    device = torch.device(device_str)
    
    # Initialize MLflow in the subprocess for real-time logging
    if mlflow_info:
        mlflow.set_experiment(mlflow_info['experiment_name'])
        # Start a fresh run for this trial. 
        # Note: True cross-process nesting is tricky, so we'll use a tag to link them.
        run_context = mlflow.start_run(run_name=mlflow_info.get('run_name'))
    else:
        run_context = nullcontext()

    with run_context:
        if mlflow_info:
            mlflow.log_params(trial_params)
            if mlflow_info.get('parent_run_id'):
                mlflow.set_tag("mlflow.parentRunId", mlflow_info['parent_run_id'])

        # Setup Dataset & Loader
        train_transform = RandomHashiAugment(stretch_prob=0.5, max_stretch=3)
        
        train_dataset = HashiDataset(
            root=Path(config['data']['root_dir']),
            split='train',
            limit=config['data'].get('limit'),
            use_degree=config['model']['use_degree'],
            use_meta_node=config['model']['use_meta_node'],
            use_row_col_meta=config['model']['use_row_col_meta'],
            use_meta_mesh=config['model']['use_meta_mesh'],
            use_meta_row_col_edges=config['model']['use_meta_row_col_edges'],
            use_distance=config['model']['use_distance'],
            use_edge_labels_as_features=config['model']['use_edge_labels_as_features'],
            use_closeness_centrality=config['model']['use_closeness_centrality'],
            use_conflict_edges=config['model']['use_conflict_edges'],
            transform=train_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_with_conflicts
        )

        # Initialize Model
        edge_dim = 3
        if config['model']['use_conflict_edges']:
            edge_dim += 1
        if config['model']['use_meta_mesh']:
            edge_dim += 1
        if config['model']['use_meta_row_col_edges']:
            edge_dim += 1
        if config['model']['use_edge_labels_as_features']:
            edge_dim += 2

        model = TransformerEdgeClassifier(
            node_embedding_dim=config['model']['node_embedding_dim'],
            hidden_channels=config['model']['hidden_channels'],
            num_layers=config['model']['num_layers'],
            heads=config['model'].get('heads', 4),
            dropout=config['model'].get('dropout', 0.25),
            use_degree=config['model']['use_degree'],
            use_meta_node=config['model']['use_meta_node'],
            use_row_col_meta=config['model']['use_row_col_meta'],
            edge_dim=edge_dim,
            use_closeness=config['model']['use_closeness_centrality'],
            use_verification_head=config['model']['use_verification_head']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        
        # Training Loop
        epochs = config['training']['epochs']
        masking_config = config['training']['masking']
        
        best_acc = 0.0
        epoch_accs = []
        for epoch in range(1, epochs + 1):
            m_rate = get_masking_rate(epoch, masking_config, epochs)
            
            metrics = run_epoch(
                model, train_loader, device,
                training=True,
                optimizer=optimizer,
                masking_rate=m_rate,
                loss_weights=config['training']['loss_weights'],
                use_verification=config['model']['use_verification_head'],
                accumulation_steps=config['training'].get('accumulation_steps', 1)
            )
            
            if mlflow_info:
                mlflow.log_metrics({
                    "train_acc": metrics.accuracy,
                    "train_loss": metrics.loss,
                    "masking_rate": m_rate
                }, step=epoch)

            epoch_accs.append((epoch, metrics.accuracy, metrics.loss, m_rate))
            best_acc = max(best_acc, metrics.accuracy)

    # Return results (will be serialized back to parent process)
    return best_acc, epoch_accs


def objective(trial: optuna.Trial, base_config: Dict[str, Any], tune_config: Dict[str, Any], device: torch.device, parent_run_id: str) -> float:
    # 1. Build trial config by lifting from base and applying suggestions
    config = yaml.safe_load(yaml.dump(base_config)) # Deep copy
    
    ss = tune_config.get('search_space', {})
    for path, spec in ss.items():
        val = get_suggestion(trial, path, spec)
        set_nested_value(config, path, val)
    
    device_str = str(device)
    
    # Pack MLflow info to pass to subprocess
    mlflow_info = {
        'experiment_name': tune_config['study_name'],
        'run_name': f"trial_{trial.number}",
        'parent_run_id': parent_run_id
    }
    
    # 2. Run trial in subprocess for clean MPS memory
    ctx = mp.get_context('spawn')
    with ctx.Pool(1) as pool:
        result = pool.apply(run_trial_in_subprocess, (config, trial.params, device_str, mlflow_info))
    
    best_acc, epoch_accs = result
    
    # 4. Report to Optuna for pruning decisions (using results returned from subproc)
    for epoch, acc, loss, m_rate in epoch_accs:
        trial.report(acc, epoch)
    
    return best_acc


def main():
    # Force spawn for MPS compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Tune Hashi GNN using Optuna.")
    parser.add_argument("--tune_config", type=str, default="configs/tune_config.yaml")
    parser.add_argument("--device", type=str, default=None, help="Override compute device (e.g. 'cpu', 'mps')")
    args = parser.parse_args()

    # Load configurations
    with open(args.tune_config, 'r') as f:
        tune_config = yaml.safe_load(f)
    base_config = load_config(tune_config['base_config_path'])
    
    # Device priority: CLI arg > Config file > Auto
    device_str = args.device or base_config['training'].get('device', 'auto')
    device = get_device(device_str)
    print(f"Using device for tuning: {device}")

    mlflow.set_experiment(tune_config['study_name'])
    
    sampler = optuna.samplers.TPESampler(multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=tune_config['study_name'],
        direction=tune_config['direction'],
        sampler=sampler,
        pruner=pruner
    )

    with mlflow.start_run(run_name="Optuna_Study_Parent") as parent_run:
        mlflow.log_params(tune_config.get('search_space', {}))
        mlflow.log_param("override_device", args.device)
        
        parent_run_id = parent_run.info.run_id
        
        study.optimize(
            lambda trial: objective(trial, base_config, tune_config, device, parent_run_id),
            n_trials=tune_config['n_trials']
        )

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Acc): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()

