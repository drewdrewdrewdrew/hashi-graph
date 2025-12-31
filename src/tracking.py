"""
Unified MLflow tracking for training and tuning.
"""
import mlflow
from contextlib import contextmanager
from typing import Dict, Any, Optional, Literal

class MLflowTracker:
    def __init__(self, mode: Literal["train", "tune"], experiment_name: str):
        self.mode = mode
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.active_run = None

    @contextmanager
    def start_parent_run(self, run_name: str, params: Optional[Dict[str, Any]] = None):
        """Start a parent run (for tuning) or a single run (for training)."""
        with mlflow.start_run(run_name=run_name) as run:
            self.active_run = run
            if params:
                mlflow.log_params(params)
            yield run

    @contextmanager
    def start_trial_run(self, trial_num: int, params: Dict[str, Any], parent_run_id: Optional[str] = None):
        """Start a trial run nested under the parent run."""
        run_name = f"trial_{trial_num}"
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            mlflow.log_params(params)
            if parent_run_id:
                mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            yield run

    def log_epoch(self, metrics: Any, step: int, prefix: str = ""):
        """Log metrics for an epoch."""
        log_data = {
            f"{prefix}loss": metrics.loss,
            f"{prefix}acc": metrics.accuracy,
            f"{prefix}perfect_acc": metrics.perfect_accuracy,
            f"{prefix}ce_loss": metrics.ce_loss,
            f"{prefix}degree_loss": metrics.degree_loss,
            f"{prefix}crossing_loss": metrics.crossing_loss,
            f"{prefix}verify_loss": metrics.verify_loss,
            f"{prefix}verify_balanced_acc": metrics.verify_balanced_acc,
        }
        mlflow.log_metrics(log_data, step=step)
