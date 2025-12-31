"""
Main script for training and evaluating the GNN model for edge classification.
"""
import argparse
import datetime
import os
import platform
from pathlib import Path
import torch
import mlflow

from .engine import (
    TrainingEngine, get_masking_rate, EarlyStopper
)
from .tracking import MLflowTracker
from .train_utils import save_config_to_model_dir
from .utils import load_config, get_device, clear_memory_cache, flatten_config
from .data import RandomHashiAugment

def optimize_cpu_threading(device: torch.device, train_config: dict):
    """CPU-specific optimizations for Apple Silicon and Intel Macs."""
    if device.type == 'cpu':
        machine = platform.machine()
        if machine == 'arm64':  # Apple Silicon
            num_cores = 11  # M3 Pro has 11 cores
            torch.set_num_threads(num_cores)
            torch.set_num_interop_threads(1)
            os.environ['OMP_NUM_THREADS'] = str(num_cores)
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_cores)
            train_config['num_workers'] = 4
            print(f"Apple Silicon ({machine}) optimized: {num_cores} threads")
        else:
            torch.set_num_threads(4)
            os.environ['OMP_NUM_THREADS'] = '4'
            print(f"Intel CPU ({machine}) optimized: 4 threads")

def print_metrics_table(epoch, m_rate, train_metrics, val_metrics=None):
    """Print a detailed metrics table for the current epoch."""
    train_str = ' | '.join(f"{val:7.4f}" for val in train_metrics.to_tuple())
    
    print(f"\nEpoch: {epoch:03d} | Mask: {m_rate:.4f}")
    print("       |                     Losses                      |               Accuracies              |")
    print("       |  Total  |   CE    |   Deg   |  Cross  |  VerL   |  Edge   |  Perf   |  VerBA  |  VerP   |  VerN   |")
    print("-------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|")
    print(f"Train  | {train_str} |")
    
    if val_metrics:
        val_str = ' | '.join(f"{val:7.4f}" for val in val_metrics.to_tuple())
        print(f"Val    | {val_str} |")

def main() -> None:
    """Train the Hashi graph model based on the configuration."""
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Train a GNN for Hashi edge classification.")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to the configuration file.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override compute device (e.g. 'cpu', 'mps')")
    args = parser.parse_args()

    config = load_config(args.config)
    device_str = args.device or config['training'].get('device', 'auto')
    device = get_device(device_str)
    print(f"Using device: {device}")

    # Apply CPU optimizations
    optimize_cpu_threading(device, config['training'])

    # Setup Training Engine
    engine = TrainingEngine(config, device)
    tracker = MLflowTracker(mode="train", experiment_name="Hashi Graph GNN")
    
    # Setup Augmentations
    train_transform = None
    aug_config = config['training'].get('augmentation', {})
    if aug_config.get('enabled', True):
        train_transform = RandomHashiAugment(
            stretch_prob=aug_config.get('stretch_prob', 0.5),
            max_stretch=aug_config.get('max_stretch', 3)
        )

    # Setup DataLoaders
    train_loader = engine.create_dataloader(split='train', transform=train_transform, use_cache=False)
    val_loader = engine.create_dataloader(split='val', use_cache=False)

    # Initialize model
    model = engine.create_model()
    print(f"Initialized {config['model']['type'].upper()} model")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("models") / f"model_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_config_to_model_dir(config, str(model_dir / "model.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    early_stopper = EarlyStopper(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta']
    )

    with tracker.start_parent_run(run_name=f"train_{timestamp}", params=flatten_config(config)):
        best_val_loss = float('inf')
        epochs = config['training']['epochs']
        eval_interval = config['training'].get('eval_interval', 1)
        accumulation_steps = config['training'].get('accumulation_steps', 1)

        for epoch in range(1, epochs + 1):
            m_rate = get_masking_rate(epoch, config['training']['masking'], epochs)
            
            train_metrics = engine.run_epoch(
                model, train_loader, training=True, 
                optimizer=optimizer, masking_rate=m_rate, 
                accumulation_steps=accumulation_steps
            )
            clear_memory_cache(device)
            tracker.log_epoch(train_metrics, step=epoch, prefix="train_")

            val_metrics = None
            if epoch % eval_interval == 0:
                val_metrics = engine.run_epoch(
                    model, val_loader, training=False, masking_rate=1.0
                )
                clear_memory_cache(device)
                tracker.log_epoch(val_metrics, step=epoch, prefix="val_")

                if val_metrics.loss < best_val_loss:
                    best_val_loss = val_metrics.loss
                    torch.save(model.state_dict(), model_dir / "best_model.pt")
                    print(f"  -> New best model saved to {model_dir / 'best_model.pt'}")
                    mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)

                if early_stopper.early_stop(val_metrics.loss):
                    print("Early stopping triggered.")
                    break
            
            # Print table every epoch
            print_metrics_table(epoch, m_rate, train_metrics, val_metrics)

if __name__ == "__main__":
    main()
