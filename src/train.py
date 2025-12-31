"""
Main script for training and evaluating the GNN model for edge classification.
"""
import argparse
import datetime
import os
import platform
from pathlib import Path
import torch

from .engine import Trainer
from .callbacks import (
    MLflowCallback, CheckpointCallback, PrintMetricsCallback
)
from .train_utils import save_config_to_model_dir
from .utils import load_config, get_device
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

    # Setup Setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("models") / f"model_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config immediately
    save_config_to_model_dir(config, str(model_dir / "model.pt"))

    # Setup Augmentations
    train_transform = None
    aug_config = config['training'].get('augmentation', {})
    if aug_config.get('enabled', True):
        train_transform = RandomHashiAugment(
            stretch_prob=aug_config.get('stretch_prob', 0.5),
            max_stretch=aug_config.get('max_stretch', 3)
        )

    # Define Callbacks
    callbacks = [
        MLflowCallback(
            experiment_name="Hashi Graph GNN",
            run_name=f"train_{timestamp}",
            params=config
        ),
        CheckpointCallback(model_dir=model_dir),
        PrintMetricsCallback()
    ]

    # Initialize Trainer
    trainer = Trainer(config, device, callbacks=callbacks)
    
    # Start Training
    trainer.train(train_transform=train_transform)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
