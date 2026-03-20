"""Entry point for REINFORCE / HashiEnv RL training (``training.mode: rl``)."""

import argparse
import datetime
import os
from pathlib import Path
import platform

import torch

from .callbacks import CheckpointCallback, MLflowCallback, PrintMetricsCallback
from .data import RandomHashiAugment
from .engine import Trainer
from .utils.common import get_device, load_config
from .utils.train_utils import save_config_to_model_dir


def optimize_cpu_threading(device_str: str, train_config: dict) -> None:
    """CPU-specific optimizations for Apple Silicon and Intel Macs."""
    is_cpu = device_str == "cpu" or (
        device_str == "auto" and not torch.cuda.is_available()
    )

    if is_cpu or platform.system() == "Darwin":
        machine = platform.machine()
        if machine == "arm64":
            try:
                num_cores = 11
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
    """Train the RL policy from a config with ``training.mode: rl``."""
    torch.multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Train Hashi RL (REINFORCE)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl_sequential.yaml",
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
    mode = config["training"].get("mode", "one-shot").lower()
    if mode != "rl":
        msg = f"training.mode must be 'rl' for this script, got {mode!r}"
        raise ValueError(msg)

    device_str = args.device or config["training"].get("device", "auto")
    optimize_cpu_threading(device_str, config["training"])
    device = get_device(device_str)
    print(f"Using device: {device}")

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    model_dir = Path("models") / f"model_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_config_to_model_dir(config, str(model_dir / "model.pt"))

    train_transform = None
    aug_config = config["training"].get("augmentation", {})
    if aug_config.get("enabled", True):
        train_transform = RandomHashiAugment(
            stretch_prob=aug_config.get("stretch_prob", 0.5),
            max_stretch=aug_config.get("max_stretch", 3),
        )

    callbacks = [
        MLflowCallback(
            experiment_name="Hashi Graph GNN",
            run_name=f"train_rl_{timestamp}",
            params=config,
        ),
        CheckpointCallback(model_dir=model_dir),
        PrintMetricsCallback(),
    ]

    trainer = Trainer(config, device, callbacks=callbacks)
    trainer.train(train_transform=train_transform)


if __name__ == "__main__":
    main()
