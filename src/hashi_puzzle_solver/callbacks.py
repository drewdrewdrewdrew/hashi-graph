"""Callbacks for GNN training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlflow
import torch

from .train_utils import save_config_to_model_dir

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
        epoch: int,
        _train_metrics: EpochMetrics,
        _val_metrics: EpochMetrics | None,
        _full_rollout_metrics: dict[str, Any] | None,
    ) -> None:
        """Save checkpoint at the end of each epoch."""
        model_path = str(self.model_dir / f"model_epoch_{epoch}.pt")
        torch.save(trainer.model.state_dict(), model_path)
        # Also save as latest
        torch.save(trainer.model.state_dict(), str(self.model_dir / "model.pt"))
        save_config_to_model_dir(trainer.config, str(self.model_dir / "model.pt"))

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

    def on_train_start(self, _trainer: Trainer) -> None:
        """Initialize MLflow run."""
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_params(self.params)

    def on_epoch_start(self, _trainer: Trainer, _epoch: int) -> None:
        """Execute logic when an epoch starts."""

    def on_epoch_end(
        self,
        _trainer: Trainer,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics | None,
        _full_rollout_metrics: dict[str, Any] | None,
    ) -> None:
        """Log metrics to MLflow."""
        metrics = {
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "train_perfect_acc": train_metrics.perfect_accuracy,
        }
        if val_metrics:
            metrics.update(
                {
                    "val_loss": val_metrics.loss,
                    "val_acc": val_metrics.accuracy,
                    "val_perfect_acc": val_metrics.perfect_accuracy,
                }
            )

        # Log AR metrics if present
        if hasattr(train_metrics, "ar_precision"):
            metrics.update(
                {
                    "train_ar_precision": train_metrics.ar_precision,
                    "train_ar_msuf": train_metrics.ar_msuf,
                    "train_ar_aced_rate": train_metrics.ar_aced_rate,
                    "train_ar_avg_rollouts_aced": train_metrics.ar_avg_rollouts_aced,
                    "train_ar_puzzle_aced_rate": train_metrics.ar_puzzle_aced_rate,
                }
            )
        if val_metrics and hasattr(val_metrics, "ar_precision"):
            metrics.update(
                {
                    "val_ar_precision": val_metrics.ar_precision,
                    "val_ar_msuf": val_metrics.ar_msuf,
                    "val_ar_aced_rate": val_metrics.ar_aced_rate,
                    "val_ar_avg_rollouts_aced": val_metrics.ar_avg_rollouts_aced,
                    "val_ar_puzzle_aced_rate": val_metrics.ar_puzzle_aced_rate,
                }
            )

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
        _full_rollout_metrics: dict[str, Any] | None,
    ) -> None:
        """Print metrics table for the current epoch."""
        mode = trainer.config["training"].get("mode", "one-shot").lower()
        if mode == "ar":
            print(f"\nEpoch: {epoch:03d} | Mode: AR")
            print("       |  Loss   |   Acc   |   MSUF  | AvgAcd | PuzAcd |")
            print("-------|---------|---------|---------|--------|--------|")
            print(
                f"Train  | {train_metrics.loss:7.4f} | "
                f"{train_metrics.ar_precision:7.4f} | "
                f"{train_metrics.ar_msuf:7.4f} | "
                f"{train_metrics.ar_avg_rollouts_aced:6.4f} | "
                f"{train_metrics.ar_puzzle_aced_rate:6.4f} |"
            )
            if val_metrics:
                print(
                    f"Val    | {val_metrics.loss:7.4f} | "
                    f"{val_metrics.ar_precision:7.4f} | "
                    f"{val_metrics.ar_msuf:7.4f} | "
                    f"{val_metrics.ar_avg_rollouts_aced:6.4f} | "
                    f"{val_metrics.ar_puzzle_aced_rate:6.4f} |"
                )
            return

        rate = trainer.current_masking_rate
        print(f"\nEpoch: {epoch:03d} | Rate: {rate:.4f}")
        print("       |                     Losses                      |                    Accuracies                   |")
        print("       |  Total  |   CE    |   Deg   |  Cross  |  VerL   |  Edge   |  Perf   |  VerBA  |  VerP   |  VerN   |")
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

    def on_train_end(self, _trainer: Trainer) -> None:
        """Execute logic when training ends."""
