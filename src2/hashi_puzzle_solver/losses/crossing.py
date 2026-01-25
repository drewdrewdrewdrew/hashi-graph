"""Crossing constraint loss for Hashi puzzles."""

import torch
import torch.nn.functional as func
from .base import LossModule


class CrossingLoss(LossModule):
    """
    Penalizes crossing bridges.
    If two edges cross, at most one can have a bridge.
    """

    def __init__(self, reduction: str = "mean", mode: str = "multiplicative"):
        self.reduction = reduction
        self.mode = mode

    def __call__(
        self,
        logits: torch.Tensor,
        edge_conflict_index: torch.Tensor | None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute crossing loss."""
        if edge_conflict_index is None or edge_conflict_index.size(1) == 0:
            return torch.tensor(0.0, device=logits.device)

        e1_indices = edge_conflict_index[0]
        e2_indices = edge_conflict_index[1]

        probs = func.softmax(logits, dim=-1)
        # Probability that a bridge exists (label 1 or 2)
        bridge_exists_prob = probs[:, 1] + probs[:, 2]

        prob1 = bridge_exists_prob[e1_indices]
        prob2 = bridge_exists_prob[e2_indices]

        if self.mode == "multiplicative":
            crossing_losses = prob1 * prob2
        elif self.mode == "max_product":
            max_logit1 = logits[e1_indices].max(dim=-1).values
            max_logit2 = logits[e2_indices].max(dim=-1).values
            crossing_losses = func.relu(max_logit1) * func.relu(max_logit2)
        else:
            msg = f"Unknown mode: {self.mode}"
            raise ValueError(msg)

        if self.reduction == "mean":
            return crossing_losses.mean()
        if self.reduction == "sum":
            return crossing_losses.sum()
        
        msg = f"Unknown reduction: {self.reduction}"
        raise ValueError(msg)
