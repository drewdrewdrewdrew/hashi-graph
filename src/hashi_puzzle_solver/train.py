"""Main script for training and evaluating the GNN model for edge classification."""

import argparse
import datetime
import os
from pathlib import Path
import platform

import torch

from .callbacks import CheckpointCallback, MLflowCallback, PrintMetricsCallback
from .data import RandomHashiAugment
from .engine import Trainer
from .train_utils import save_config_to_model_dir
from .utils import get_device, load_config


def optimize_cpu_threading(device_str: str, train_config: dict) -> None:
    """CPU-specific optimizations for Apple Silicon and Intel Macs."""
    # Check if we are likely using CPU
    # We can't check device.type yet because creating the device might init the backend
    is_cpu = device_str == "cpu" or (
        device_str == "auto" and not torch.cuda.is_available()
    )

    # On Apple Silicon, we want to force CPU optimization even if 'mps' is available
    # if the user specifically asked for 'cpu' or if 'auto' falls back to it.
    # Note: torch.backends.mps.is_available() checks for MPS

    if is_cpu or platform.system() == "Darwin":
        # On Mac, even with MPS, setting threads helps dataloader performance
        machine = platform.machine()
        if machine == "arm64":  # Apple Silicon
            try:
                num_cores = 11  # M3 Pro has 11 cores
                torch.set_num_threads(num_cores)
                torch.set_num_interop_threads(1)
                os.environ["OMP_NUM_THREADS"] = str(num_cores)
                os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cores)
                train_config["num_workers"] = 4
                print(f"Apple Silicon ({machine}) optimized: {num_cores} threads")
            except RuntimeError as e:
                print(
                    f"Warning: Could not set threading options "
                    f"(backend already initialized): {e}",
                )
        else:
            try:
                torch.set_num_threads(4)
                os.environ["OMP_NUM_THREADS"] = "4"
                print(f"Intel CPU ({machine}) optimized: 4 threads")
            except RuntimeError as e:
                print(f"Warning: Could not set threading options: {e}")


def main() -> None:
    """Train the Hashi graph model based on the configuration."""
    torch.multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Train GNN for Hashi")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (e.g. 'cpu', 'mps')",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device_str = args.device or config["training"].get("device", "auto")

    # Apply CPU optimizations BEFORE getting device
    optimize_cpu_threading(device_str, config["training"])

    device = get_device(device_str)
    print(f"Using device: {device}")

    # Setup Setup
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    model_dir = Path("models") / f"model_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save config immediately
    save_config_to_model_dir(config, str(model_dir / "model.pt"))

    # Setup Augmentations
    train_transform = None
    aug_config = config["training"].get("augmentation", {})
    if aug_config.get("enabled", True):
        train_transform = RandomHashiAugment(
            stretch_prob=aug_config.get("stretch_prob", 0.5),
            max_stretch=aug_config.get("max_stretch", 3),
        )

    # Define Callbacks
    callbacks = [
        MLflowCallback(
            experiment_name="Hashi Graph GNN",
            run_name=f"train_{timestamp}",
            params=config,
        ),
        CheckpointCallback(model_dir=model_dir),
        PrintMetricsCallback(),
    ]

    # Initialize Trainer
    trainer = Trainer(config, device, callbacks=callbacks)

    # Start Training
    trainer.train(train_transform=train_transform)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
