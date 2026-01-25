"""Loss modules for Hashi Puzzle Solver."""

from .base import LossModule
from .degree import DegreeLoss
from .crossing import CrossingLoss
from .calculator import HashiLossCalculator
from .legacy import compute_combined_loss

__all__ = [
    "LossModule",
    "DegreeLoss",
    "CrossingLoss",
    "HashiLossCalculator",
    "compute_combined_loss",
]
