"""Reverse GNN backbone for rev-reasoning (REVG-01, REVG-02, REVG-03)."""

from __future__ import annotations

import torch
from torch.nn import Linear

from hashi_puzzle_solver.models.backbone import GraphBackbone


class ReverseBackbone(torch.nn.Module):
    """Backbone that runs on reversed edges.

    Accepts the same inputs as GraphBackbone but flips edge direction before
    the message-passing call. Optionally holds a linear projection layer for
    compressing concatenated [forward_h, reverse_h] to hidden_channels (used
    by Phase 5 integration).

    Args:
        forward_backbone: The forward-pass GraphBackbone. Used as a template
            for mirror construction (separate_weights=True) or as the shared
            weight source (separate_weights=False).
        hidden_channels: Target output dimension for the optional projection.
        separate_weights: If True, own a new independent GraphBackbone.
            If False, share the forward_backbone's weights without registering
            it as a submodule (avoids double-counting in optimizer).
        project_embeddings: If True, register a Linear(2 * final_dim,
            hidden_channels) layer for use by Phase 5 integration.

    Note on edge_attr directionality: edge_attr is passed as-is (not flipped).
        If edge attributes are directional, Phase 5 integration can address this.
    """

    def __init__(
        self,
        forward_backbone: GraphBackbone,
        hidden_channels: int,
        separate_weights: bool = True,
        project_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.separate_weights = separate_weights
        self.project_embeddings = project_embeddings

        if separate_weights:
            # Own independent parameters — mirror constructor args from forward backbone
            self.backbone = GraphBackbone(
                node_input_dim=forward_backbone.convs[0].in_channels,
                hidden_channels=hidden_channels,
                num_layers=len(forward_backbone.convs),
                heads=forward_backbone.convs[0].heads,
                dropout=forward_backbone.dropout,
                edge_dim=forward_backbone.convs[0].edge_dim,
                gnn_type=forward_backbone.gnn_type,
            )
        else:
            # Share weights without registering as submodule.
            # object.__setattr__ bypasses nn.Module.__setattr__ so the
            # forward backbone is NOT added to self._modules.
            object.__setattr__(self, "_shared_backbone", forward_backbone)

        # final_dim matches the backbone being used
        if separate_weights:
            self.final_dim = self.backbone.final_dim
        else:
            self.final_dim = forward_backbone.final_dim

        if project_embeddings:
            # Projection for Phase 5: compresses [fwd_h || rev_h] -> hidden_channels
            self.projection = Linear(2 * self.final_dim, hidden_channels)

    def _get_backbone(self) -> GraphBackbone:
        if self.separate_weights:
            return self.backbone
        return self._shared_backbone  # type: ignore[attr-defined]

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run backbone on reversed edges. Returns reverse node embeddings.

        Note: Concatenation of forward + reverse and optional projection
        are handled by Phase 5 (HashiGraphModel.forward).
        """
        rev_edge_index = edge_index.flip(0)
        return self._get_backbone()(h, rev_edge_index, edge_attr=edge_attr)
