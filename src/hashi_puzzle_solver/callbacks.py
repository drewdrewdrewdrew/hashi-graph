"""Callbacks for GNN training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlflow
import torch

from .train_utils import save_config_to_model_dir
from .utils import flatten_config

if TYPE_CHECKING:
    from pathlib import Path

    from .engine import EpochMetrics, Trainer


class CheckpointCallback:
    """Callback to save model checkpoints."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir

    def on_train_start(self, _trainer: Trainer) -> None:
        """Execute logic when training starts."""

    def on_epoch_start(self, _trainer: Trainer, _epoch: int) -> None:
        """Execute logic when an epoch starts."""

    def on_epoch_end(
        self,
        trainer: Trainer,
        _epoch: int,
        _train_metrics: EpochMetrics,
        _val_metrics: EpochMetrics | None,
        _full_rollout_metrics: dict[str, Any] | None,
    ) -> None:
        """Save checkpoint at the end of each epoch."""
        latest_path = str(self.model_dir / "model_latest.pt")
        torch.save(trainer.model.state_dict(), latest_path)
        # Also save config quietly
        save_config_to_model_dir(trainer.config, latest_path)

    def on_train_end(self, _trainer: Trainer) -> None:
        """Execute logic when training ends."""


class MLflowCallback:
    """Callback for MLflow logging."""

    def __init__(
        self, experiment_name: str, run_name: str, params: dict[str, Any]
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.params = params

    def on_train_start(self, trainer: Trainer) -> None:
        """Initialize MLflow run and log all parameters."""
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

        # Log flattened config as params
        flat_params = flatten_config(trainer.config)
        mlflow.log_params(flat_params)

    def on_epoch_start(self, _trainer: Trainer, _epoch: int) -> None:
        """Execute logic when an epoch starts."""

    def on_epoch_end(
        self,
        _trainer: Trainer,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics | None,
        full_rollout_metrics: dict[str, Any] | None,
    ) -> None:
        """Log metrics to MLflow."""
        metrics = {
            "train_loss": train_metrics.loss,
            "train_ce_loss": train_metrics.ce_loss,
            "train_degree_loss": train_metrics.degree_loss,
            "train_crossing_loss": train_metrics.crossing_loss,
            "train_verify_loss": train_metrics.verify_loss,
            "train_acc": train_metrics.accuracy,
            "train_perfect_acc": train_metrics.perfect_accuracy,
        }
        if val_metrics:
            metrics.update(
                {
                    "val_loss": val_metrics.loss,
                    "val_ce_loss": val_metrics.ce_loss,
                    "val_degree_loss": val_metrics.degree_loss,
                    "val_crossing_loss": val_metrics.crossing_loss,
                    "val_verify_loss": val_metrics.verify_loss,
                    "val_acc": val_metrics.accuracy,
                    "val_perfect_acc": val_metrics.perfect_accuracy,
                }
            )

        if full_rollout_metrics:
            for k, v in full_rollout_metrics.items():
                metrics[f"rollout_{k}"] = v

        mlflow.log_metrics(metrics, step=epoch)

    def on_train_end(self, _trainer: Trainer) -> None:
        """End MLflow run."""
        mlflow.end_run()


class PrintMetricsCallback:
    """Callback to print metrics to console."""

    def on_train_start(self, _trainer: Trainer) -> None:
        """Execute logic when training starts."""

    def on_epoch_start(self, _trainer: Trainer, _epoch: int) -> None:
        """Execute logic when an epoch starts."""

    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics | None,
        full_rollout_metrics: dict[str, Any] | None,
    ) -> None:
        """Print metrics table for the current epoch."""
        mode = trainer.config["training"].get("mode", "one-shot").lower()
        rate = getattr(trainer, "current_masking_rate", 1.0)

        print(f"\nEpoch: {epoch:03d} | Mode: {mode} | Rate: {rate:.4f}")

        print(
            "       |                     Losses                      "
            "|                    Accuracies                   |"
        )
        print(
            "       |  Total  |   CE    |   Deg   |  Cross  |  VerL   "
            "|  Edge   |  Perf   |  VerBA  |  VerP   |  VerN   |"
        )
        print("-------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|")

        train_fmt = (
            f"Train  | {train_metrics.loss:7.4f} | "
            f"{train_metrics.ce_loss:7.4f} | "
            f"{train_metrics.degree_loss:7.4f} | "
            f"{train_metrics.crossing_loss:7.4f} | "
            f"{train_metrics.verify_loss:7.4f} | "
            f"{train_metrics.accuracy:7.4f} | "
            f"{train_metrics.perfect_accuracy:7.4f} | "
            f"{train_metrics.verify_balanced_acc:7.4f} | "
            f"{train_metrics.verify_recall_pos:7.4f} | "
            f"{train_metrics.verify_recall_neg:7.4f} |"
        )
        print(train_fmt)

        if val_metrics:
            val_fmt = (
                f"Val    | {val_metrics.loss:7.4f} | "
                f"{val_metrics.ce_loss:7.4f} | "
                f"{val_metrics.degree_loss:7.4f} | "
                f"{val_metrics.crossing_loss:7.4f} | "
                f"{val_metrics.verify_loss:7.4f} | "
                f"{val_metrics.accuracy:7.4f} | "
                f"{val_metrics.perfect_accuracy:7.4f} | "
                f"{val_metrics.verify_balanced_acc:7.4f} | "
                f"{val_metrics.verify_recall_pos:7.4f} | "
                f"{val_metrics.verify_recall_neg:7.4f} |"
            )
            print(val_fmt)

        if full_rollout_metrics:
            print("\nIterative Rollout Validation Metrics:")
            rollout_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in full_rollout_metrics.items()
            )
            print(f"       | {rollout_str}")

    def on_train_end(self, _trainer: Trainer) -> None:
        """Execute logic when training ends."""
