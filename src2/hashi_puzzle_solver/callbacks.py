"""Callbacks for GNN training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlflow
import torch

from .utils.common import flatten_config
from .utils.train_utils import save_config_to_model_dir

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
        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        try:
            import requests
            resp = requests.get("http://127.0.0.1:5001/health", timeout=2)
            resp.raise_for_status()
        except Exception:
            print("WARNING: MLflow server not reachable at http://127.0.0.1:5000 — disabling MLflow logging")
            self._disabled = True
            return

        self._disabled = False
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

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
        if getattr(self, "_disabled", False):
            return
        metrics = {
            "train_loss": train_metrics.loss,
            "train_ce_loss": train_metrics.ce_loss,
            "train_degree_loss": train_metrics.degree_loss,
            "train_crossing_loss": train_metrics.crossing_loss,
            "train_verify_loss": train_metrics.verify_loss,
            "train_verify_balanced_acc": train_metrics.verify_balanced_acc,
            "train_verify_recall_pos": train_metrics.verify_recall_pos,
            "train_verify_recall_neg": train_metrics.verify_recall_neg,
            "train_noise_loss": train_metrics.noise_loss,
            "train_sigma_loss": train_metrics.sigma_loss,
            "train_alpha_loss": train_metrics.alpha_loss,
            "train_acc": train_metrics.accuracy,
            "train_perfect_acc": train_metrics.perfect_accuracy,
            "train_residual_mse": train_metrics.residual_mse,
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
                    "val_noise_loss": val_metrics.noise_loss,
                    "val_sigma_loss": val_metrics.sigma_loss,
                    "val_alpha_loss": val_metrics.alpha_loss,
                    "val_acc": val_metrics.accuracy,
                    "val_perfect_acc": val_metrics.perfect_accuracy,
                    "val_residual_mse": val_metrics.residual_mse,
                }
            )

        if full_rollout_metrics:
            for k, v in full_rollout_metrics.items():
                metrics[f"rollout_{k}"] = v

        mlflow.log_metrics(metrics, step=epoch)

    def on_train_end(self, _trainer: Trainer) -> None:
        """End MLflow run."""
        if getattr(self, "_disabled", False):
            return
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
        if mode == "rl":
            # Only print when we have evaluation metrics (like eval_interval in standard training)
            if not full_rollout_metrics:
                return
            
            print(f"\nEpoch: {epoch:03d} | Mode: rl | Train Loss: {train_metrics.loss:.4f}")
            print("\nRollout Validation Metrics:")
            
            # Format rollout metrics in a clean table
            print("       | Perf Acc | Edge Acc | Avg Ret  | Ep Len   | Solve Len | Oracle Fail | Cap Fail | Cross Fail | Dead End | Max Steps |")
            print("-------|----------|----------|----------|----------|-----------|-------------|----------|------------|----------|-----------|")
            
            perf_acc = full_rollout_metrics.get("perfect_accuracy", 0.0)
            edge_acc = full_rollout_metrics.get("edge_acc", 0.0)
            avg_ret = full_rollout_metrics.get("avg_return", 0.0)
            ep_len = full_rollout_metrics.get("avg_episode_length", 0.0)
            solve_len = full_rollout_metrics.get("avg_solve_length", 0.0)
            oracle_fail = full_rollout_metrics.get("oracle_failure_rate", 0.0)
            cap_fail = full_rollout_metrics.get("capacity_failure_rate", 0.0)
            cross_fail = full_rollout_metrics.get("crossing_failure_rate", 0.0)
            dead_end = full_rollout_metrics.get("dead_end_unsolved_rate", 0.0)
            max_steps = full_rollout_metrics.get("max_steps_rate", 0.0)
            
            print(f"Val    | {perf_acc:8.4f} | {edge_acc:8.4f} | {avg_ret:8.4f} | {ep_len:8.2f} | {solve_len:9.2f} | {oracle_fail:11.4f} | {cap_fail:8.4f} | {cross_fail:10.4f} | {dead_end:8.4f} | {max_steps:9.4f} |")
            return

        rate = getattr(trainer, "current_masking_rate", 1.0)
        use_verify = trainer.config["model"].get("use_verification_head", False)

        print(f"\nEpoch: {epoch:03d} | Mode: {mode} | Rate: {rate:.4f}")

        def fmt_val(val: float) -> str:
            return f" {val:7.4f} |"

        # Build dynamic headers
        # Labels and their corresponding metric values
        loss_cols = [
            ("Total", "loss"),
        ]
        
        if mode == "residual":
            loss_cols.append(("ResMSE", "residual_mse"))
        else:
            loss_cols.append(("CE", "ce_loss"))
        
        loss_cols.extend([
            ("Deg", "degree_loss"),
            ("Cross", "crossing_loss"),
        ])
        
        if mode not in ["flow-blind", "residual"]:
            loss_cols.append(("NoiseL", "noise_loss"))

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
        def make_row(cols: list[tuple[str, str]]) -> str:
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
        def make_metrics_row(label: str, metrics: EpochMetrics) -> str:
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
