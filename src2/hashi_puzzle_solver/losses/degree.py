"""Degree violation loss for Hashi puzzles."""

import torch
import torch.nn.functional as func
from torch_geometric.utils import scatter
from .base import LossModule


class DegreeLoss(LossModule):
    """
    Penalizes violations of the island capacity constraint.
    Sum of bridges for each island should equal its capacity.
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def __call__(
        self,
        logits: torch.Tensor,
        edge_index: torch.Tensor,
        node_capacities: torch.Tensor,
        edge_mask: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute degree violation loss."""
        # Get soft bridge values: E[label] = 1*P(1) + 2*P(2)
        probs = func.softmax(logits, dim=-1)
        bridge_values = probs[:, 1] * 1.0 + probs[:, 2] * 2.0
        
        # Only original puzzle edges contribute to degree
        bridge_values_masked = bridge_values * edge_mask.float()
        
        # Sum bridge values per node
        src_nodes = edge_index[0]
        predicted_degrees = scatter(
            bridge_values_masked, src_nodes, dim=0, reduce="sum"
        )
        
        # Target degrees are node capacities (only for islands 1-8)
        target_degrees = node_capacities.float()
        is_puzzle_node = (node_capacities >= 1) & (node_capacities <= 8)
        target_degrees = target_degrees * is_puzzle_node.float()
        
        # Padding if necessary
        if predicted_degrees.size(0) < target_degrees.size(0):
            padding = torch.zeros(
                target_degrees.size(0) - predicted_degrees.size(0),
                device=predicted_degrees.device,
                dtype=predicted_degrees.dtype,
            )
            predicted_degrees = torch.cat([predicted_degrees, padding])
            
        return func.mse_loss(predicted_degrees, target_degrees, reduction=self.reduction)
