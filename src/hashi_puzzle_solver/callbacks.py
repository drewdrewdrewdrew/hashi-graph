from typing import Dict, Any, Optional
import torch
from pathlib import Path
import mlflow
import optuna

class BaseCallback:
    """Interface for training callbacks."""
    def on_train_start(self, trainer: Any):
        pass

    def on_epoch_start(self, trainer: Any, epoch: int):
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, train_metrics: Any, val_metrics: Optional[Any] = None):
        pass

    def on_train_end(self, trainer: Any):
        pass

class MLflowCallback(BaseCallback):
    """Logs metrics and parameters to MLflow."""
    def __init__(self, experiment_name: str, run_name: str, params: Dict[str, Any], nested: bool = False):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.params = params
        self.nested = nested
        self.run = None

    def on_train_start(self, trainer: Any):
        # Only set experiment if no run is active to avoid interfering with parent runs
        if not mlflow.active_run():
            mlflow.set_experiment(self.experiment_name)
            print(f"MLflowCallback: Set experiment to '{self.experiment_name}'")
        else:
            print(f"MLflowCallback: Active run detected, skipping set_experiment")

        # Start the run (nested if requested)
        self.run = mlflow.start_run(run_name=self.run_name, nested=self.nested)
        print(f"MLflowCallback: Started run '{self.run_name}' (nested={self.nested}) with run_id: {self.run.info.run_id}")

        if self.params:
            from .utils import flatten_config
            flattened = flatten_config(self.params)
            mlflow.log_params(flattened)
            print(f"MLflowCallback: Logged {len(flattened)} parameters")

    def on_epoch_end(self, trainer: Any, epoch: int, train_metrics: Any, val_metrics: Optional[Any] = None):
        self._log_metrics(train_metrics, epoch, prefix="train_")
        if val_metrics:
            self._log_metrics(val_metrics, epoch, prefix="val_")

    def _log_metrics(self, metrics: Any, step: int, prefix: str):
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

    def on_train_end(self, trainer: Any):
        if self.run:
            mlflow.end_run()

class OptunaPruningCallback(BaseCallback):
    """Reports metrics to Optuna and raises TrialPruned if necessary."""
    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, trainer: Any, epoch: int, train_metrics: Any, val_metrics: Optional[Any] = None):
        if val_metrics is None:
            return

        # Map monitor string to metric value
        if self.monitor == "val_loss":
            score = val_metrics.loss
        elif self.monitor == "val_acc":
            score = val_metrics.accuracy
        else:
            raise ValueError(f"Unknown monitor metric: {self.monitor}")

        print(f"OptunaPruningCallback: Trial {self.trial.number} epoch {epoch} - reporting {self.monitor} = {score}")
        self.trial.report(score, epoch)
        if self.trial.should_prune():
            print(f"OptunaPruningCallback: Trial {self.trial.number} pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

class CheckpointCallback(BaseCallback):
    """Saves model checkpoints based on validation loss."""
    def __init__(self, model_dir: Path, monitor: str = "val_loss", mode: str = "min"):
        self.model_dir = model_dir
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: Any, epoch: int, train_metrics: Any, val_metrics: Optional[Any] = None):
        if val_metrics is None:
            return

        if self.monitor == "val_loss":
            score = val_metrics.loss
        elif self.monitor == "val_acc":
            score = val_metrics.accuracy
        else:
            raise ValueError(f"Unknown monitor metric: {self.monitor}")

        is_best = (self.mode == "min" and score < self.best_score) or \
                  (self.mode == "max" and score > self.best_score)

        if is_best:
            self.best_score = score
            save_path = self.model_dir / "best_model.pt"
            torch.save(trainer.model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path} (score: {score:.4f})")
            if mlflow.active_run():
                mlflow.log_metric(f"best_{self.monitor}", self.best_score, step=epoch)

class PrintMetricsCallback(BaseCallback):
    """Prints a detailed metrics table for the current epoch."""
    def on_epoch_end(self, trainer: Any, epoch: int, train_metrics: Any, val_metrics: Optional[Any] = None):
        m_rate = trainer.current_masking_rate
        train_str = ' | '.join(f"{val:7.4f}" for val in train_metrics.to_tuple())
        
        print(f"\nEpoch: {epoch:03d} | Mask: {m_rate:.4f}")
        print("       |                     Losses                      |               Accuracies              |")
        print("       |  Total  |   CE    |   Deg   |  Cross  |  VerL   |  Edge   |  Perf   |  VerBA  |  VerP   |  VerN   |")
        print("-------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|")
        print(f"Train  | {train_str} |")
        
        if val_metrics:
            val_str = ' | '.join(f"{val:7.4f}" for val in val_metrics.to_tuple())
            print(f"Val    | {val_str} |")

