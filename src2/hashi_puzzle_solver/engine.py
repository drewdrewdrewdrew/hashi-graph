"""
Core training engine for Hashi GNN.

Centralizes model creation, dataset loading, and training loop components.
"""

from typing import Any

import torch
from torch.utils.data import DataLoader

from .rl.trainer import RLTrainer
from .trainers.ar import ARTrainer
from .trainers.base import EpochMetrics
from .trainers.diffusion import DiffusionTrainer
from .trainers.one_shot import OneShotTrainer


def create_trainer(
    config: dict[str, Any],
    device: torch.device,
    callbacks: list[Any] | None = None,
) -> Any:
    """Factory function to create the appropriate trainer based on config."""
    mode = config["training"].get("mode", "one-shot").lower()

    if mode == "ar":
        return ARTrainer(config, device, callbacks)
    if mode in ["diff-discrete", "diff-cont", "flow-blind"]:
        return DiffusionTrainer(config, device, callbacks)
    if mode == "rl":
        return RLTrainer(config, device, callbacks)
    return OneShotTrainer(config, device, callbacks)


class Trainer:
    """
    Facade for training Hashi GNN models.
    Now acts as a wrapper around specialized trainer classes.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        callbacks: list[Any] | None = None,
    ):
        self.config = config
        self.device = device
        self.callbacks = callbacks or []
        self._trainer = create_trainer(config, device, callbacks)

    def train(self, train_transform: Any | None = None) -> None:
        """Run the training loop using the underlying specialized trainer."""
        self._trainer.train(train_transform)

    def _setup(self, *args, **kwargs) -> None:
        """Internal setup for testing or manual control."""
        return self._trainer._setup(*args, **kwargs)

    def run_epoch_one_shot(self, *args, **kwargs) -> EpochMetrics:
        """Run a one-shot epoch (compatibility for tests)."""
        if isinstance(self._trainer, OneShotTrainer):
            # Strip model/optimizer from kwargs if they were passed (old API)
            kwargs.pop("model", None)
            kwargs.pop("optimizer", None)
            kwargs.pop("masking_rate", None)  # OneShotTrainer uses current_masking_rate
            return self._trainer.run_epoch(*args, **kwargs)
        # Fallback for other trainers if needed, or just let it fail
        raise AttributeError("Only OneShotTrainer supports run_epoch_one_shot")

    def create_dataloader(self, *args, **kwargs) -> DataLoader:
        """Delegate dataloader creation to the underlying trainer."""
        return self._trainer.create_dataloader(*args, **kwargs)

    # Maintain properties for backward compatibility with callbacks or external code
    @property
    def model(self):
        return self._trainer.model

    @property
    def optimizer(self):
        return self._trainer.optimizer

    @property
    def train_loader(self):
        return self._trainer.train_loader

    @property
    def val_loader(self):
        return self._trainer.val_loader

    @property
    def model_config(self):
        return self._trainer.model_config
