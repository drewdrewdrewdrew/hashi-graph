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
            "train_verify_balanced_acc": train_metrics.verify_balanced_acc,
            "train_verify_recall_pos": train_metrics.verify_recall_pos,
            "train_verify_recall_neg": train_metrics.verify_recall_neg,
            "train_sigma_loss": train_metrics.sigma_loss,
            "train_alpha_loss": train_metrics.alpha_loss,
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
                    "val_verify_balanced_acc": val_metrics.verify_balanced_acc,
                    "val_verify_recall_pos": val_metrics.verify_recall_pos,
                    "val_verify_recall_neg": val_metrics.verify_recall_neg,
                    "val_sigma_loss": val_metrics.sigma_loss,
                    "val_alpha_loss": val_metrics.alpha_loss,
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
        use_verify = trainer.config["model"].get("use_verification_head", False)

        print(f"\nEpoch: {epoch:03d} | Mode: {mode} | Rate: {rate:.4f}")

        def fmt_val(val: float) -> str:
            return f" {val:7.4f} |"

        # Build dynamic headers
        # Labels and their corresponding metric values
        loss_cols = [
            ("Total", "loss"),
            ("CE", "ce_loss"),
            ("Deg", "degree_loss"),
            ("Cross", "crossing_loss"),
            ("SigL", "sigma_loss"),
            ("AlpL", "alpha_loss"),
        ]
        
        verify_cols = []
        if use_verify:
            verify_cols = [
                ("VerL", "verify_loss"),
                ("VerBA", "verify_balanced_acc"),
                ("VerP", "verify_recall_pos"),
                ("VerN", "verify_recall_neg"),
            ]
            
        acc_cols = [
            ("Edge", "accuracy"),
            ("Perf", "perfect_accuracy"),
        ]

        # Assemble headers
        def make_row(cols):
            return "".join(f"  {name:<6} |" for name, _ in cols)

        header_l1 = "       |"
        header_l2 = "       |"
        
        # Losses section
        l_width = len(make_row(loss_cols))
        header_l1 += f"{'Losses':^{l_width}}|"
        header_l2 += make_row(loss_cols)
        
        # Verification section
        if use_verify:
            v_width = len(make_row(verify_cols))
            header_l1 += f"{'Verification':^{v_width}}|"
            header_l2 += make_row(verify_cols)
            
        # Accuracies section
        a_width = len(make_row(acc_cols))
        header_l1 += f"{'Accuracies':^{a_width}}|"
        header_l2 += make_row(acc_cols)
        
        header_l3 = "-" * (len(header_l2))

        # Format metrics rows
        def make_metrics_row(label, metrics):
            row = f"{label:<7}|"
            for _, attr in loss_cols:
                row += fmt_val(getattr(metrics, attr))
            if use_verify:
                for _, attr in verify_cols:
                    row += fmt_val(getattr(metrics, attr))
            for _, attr in acc_cols:
                row += fmt_val(getattr(metrics, attr))
            return row

        print(header_l1)
        print(header_l2)
        print(header_l3)
        print(make_metrics_row("Train", train_metrics))
        if val_metrics:
            print(make_metrics_row("Val", val_metrics))

        if full_rollout_metrics:
            print("\nIterative Rollout Validation Metrics:")
            rollout_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in full_rollout_metrics.items()
            )
            print(f"       | {rollout_str}")

    def on_train_end(self, _trainer: Trainer) -> None:
        """Execute logic when training ends."""
