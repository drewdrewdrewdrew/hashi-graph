"""Iterative shared-weight backbone for reasoning (REAS-01, REAS-02)."""

import torch
import torch.nn.functional as func
from torch.nn import LayerNorm, Linear
from torch_geometric.nn import TransformerConv


class IterativeBackbone(torch.nn.Module):
    """Single shared-weight TransformerConv applied K times with residual updates.

    Unlike GraphBackbone (which stacks N distinct conv layers), this module
    reuses one conv layer self.steps times. concat=False is fixed so output
    dimension equals hidden_channels, making the residual add unconditional.

    When reverse is enabled, provides reverse() for Park et al. (ICML 2024)
    fixed-point iteration (Algorithm 1) and a projection layer for combining
    forward and reverse embeddings.

    Args:
        hidden_channels: Node embedding dimension (input AND output — must match).
        steps: Number of iterative message-passing rounds (>= 1).
        heads: Attention heads. Output is averaged (concat=False), so
               final dimension is hidden_channels regardless of heads.
        dropout: Dropout probability applied after activation.
        edge_dim: Edge feature dimension, or None for no edge features.
        reverse_enabled: If True, create projection layer for [h_fwd || h_rev].
    """

    def __init__(
        self,
        hidden_channels: int,
        steps: int,
        heads: int = 1,
        dropout: float = 0.25,
        edge_dim: int | None = None,
        reverse_enabled: bool = False,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self.steps = steps
        self.dropout = dropout
        self.reverse_enabled = reverse_enabled
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

        if reverse_enabled:
            self.projection = Linear(2 * hidden_channels, hidden_channels)

    def _conv_block(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        use_dropout: bool = True,
    ) -> torch.Tensor:
        """Shared conv + norm + relu + optional dropout (the 'h(X)' in the paper)."""
        out = self.conv(h, edge_index, edge_attr=edge_attr)
        out = self.norm(out)
        out = func.relu(out)
        if use_dropout:
            out = func.dropout(out, p=self.dropout, training=self.training)
        return out

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the shared conv self.steps times with residual updates."""
        for _ in range(self.steps):
            h_in = h
            h = self._conv_block(h, edge_index, edge_attr)
            h = h + h_in
        return h

    def reverse(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        fixed_point_iters: int,
    ) -> torch.Tensor:
        """Park et al. Algorithm 1: compute f^{-1} via fixed-point iteration.

        For each of K forward steps, run M sub-iterations of
        ``h <- h - conv_block(h)`` (deterministic — no dropout).
        """
        for _ in range(self.steps):
            for _ in range(fixed_point_iters):
                h = h - self._conv_block(h, edge_index, edge_attr, use_dropout=False)
        return h

    def enforce_lipschitz(self, c: float) -> None:
        """Clamp each linear weight in self.conv so ||W||_F <= c."""
        for name in ("lin_key", "lin_query", "lin_value", "lin_edge"):
            lin = getattr(self.conv, name, None)
            if lin is None:
                continue
            w = lin.weight
            norm = w.norm(p="fro")
            if norm > c:
                w.data.mul_(c / norm)
