"""Base interface for Hashi loss modules."""

from abc import ABC, abstractmethod
import torch


class LossModule(ABC):
    """Abstract base class for individual loss components."""

    @abstractmethod
    def __call__(self, **kwargs) -> torch.Tensor:
        """Compute the loss value."""
        pass
