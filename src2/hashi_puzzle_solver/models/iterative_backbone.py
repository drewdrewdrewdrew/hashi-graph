"""Iterative shared-weight backbone for reasoning (REAS-01, REAS-02)."""

import torch
import torch.nn.functional as func
from torch.nn import LayerNorm
from torch_geometric.nn import TransformerConv


class IterativeBackbone(torch.nn.Module):
    """Single shared-weight TransformerConv applied K times with residual updates.

    Unlike GraphBackbone (which stacks N distinct conv layers), this module
    reuses one conv layer self.steps times. concat=False is fixed so output
    dimension equals hidden_channels, making the residual add unconditional.

    Args:
        hidden_channels: Node embedding dimension (input AND output — must match).
        steps: Number of iterative message-passing rounds (>= 1).
        heads: Attention heads. Output is averaged (concat=False), so
               final dimension is hidden_channels regardless of heads.
        dropout: Dropout probability applied after activation.
        edge_dim: Edge feature dimension, or None for no edge features.
    """

    def __init__(
        self,
        hidden_channels: int,
        steps: int,
        heads: int = 1,
        dropout: float = 0.25,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self.steps = steps
        self.dropout = dropout
        # concat=False: output is hidden_channels regardless of heads
        self.conv = TransformerConv(
            hidden_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=False,
        )
        self.norm = LayerNorm(hidden_channels)
        self.final_dim = hidden_channels

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the shared conv self.steps times with residual updates."""
        for _ in range(self.steps):
            h_in = h
            h = self.conv(h, edge_index, edge_attr=edge_attr)
            h = self.norm(h)
            h = func.relu(h)
            h = func.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in  # residual always valid: in_channels == out_channels
        return h
