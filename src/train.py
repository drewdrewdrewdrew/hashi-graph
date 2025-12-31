"""
Main script for training and evaluating the GNN model for edge classification.
"""
import argparse
import datetime
from pathlib import Path
import torch
import mlflow

from .engine import (
    create_model, create_dataloader, run_epoch, 
    get_masking_rate, EarlyStopper, EpochMetrics
)
from .tracking import MLflowTracker
from .train_utils import save_config_to_model_dir
from .utils import load_config, get_device, clear_memory_cache, flatten_config

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

    # Initialize tracker
    tracker = MLflowTracker(mode="train", experiment_name="Hashi Graph GNN")
    
    # Setup DataLoaders
    train_loader = create_dataloader(config, split='train', use_cache=False)
    val_loader = create_dataloader(config, split='val', use_cache=False)

    # Initialize model
    model = create_model(config['model'], device)
    print(f"Initialized {config['model']['type'].upper()} model")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("models") / f"model_{timestamp}"
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
            
            train_metrics = run_epoch(
                model, train_loader, device, config,
                training=True, optimizer=optimizer,
                masking_rate=m_rate, accumulation_steps=accumulation_steps
            )
            clear_memory_cache(device)
            tracker.log_epoch(train_metrics, step=epoch, prefix="train_")

            if epoch % eval_interval == 0:
                val_metrics = run_epoch(
                    model, val_loader, device, config,
                    training=False, masking_rate=1.0
                )
                clear_memory_cache(device)
                tracker.log_epoch(val_metrics, step=epoch, prefix="val_")

                # Print summary
                print(f"Epoch {epoch:03d} | Loss: {train_metrics.loss:.4f} | Val Loss: {val_metrics.loss:.4f} | Acc: {train_metrics.accuracy:.4f} | Val Acc: {val_metrics.accuracy:.4f}")

                if val_metrics.loss < best_val_loss:
                    best_val_loss = val_metrics.loss
                    torch.save(model.state_dict(), model_dir / "best_model.pt")
                    print(f"  -> New best model saved")
                    mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)

                if early_stopper.early_stop(val_metrics.loss):
                    print("Early stopping triggered.")
                    break

if __name__ == "__main__":
    main()
