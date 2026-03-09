"""Tests for ReverseBackbone — covers REVG-01, REVG-02, REVG-03."""

import pytest
import torch

from hashi_puzzle_solver.models.backbone import GraphBackbone
from hashi_puzzle_solver.models.reverse_backbone import ReverseBackbone


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def forward_backbone() -> GraphBackbone:
    """Minimal single-layer TransformerConv backbone, CPU float32."""
    torch.manual_seed(42)
    return GraphBackbone(
        node_input_dim=16,
        hidden_channels=16,
        num_layers=1,
        heads=1,
        gnn_type="transformer",
    )


@pytest.fixture
def graph_inputs():
    """4-node graph with 6 directed edges, CPU float32."""
    torch.manual_seed(42)
    h = torch.randn(4, 16)
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 3],
         [1, 2, 2, 3, 3, 0]],
        dtype=torch.long,
    )
    return h, edge_index


# ---------------------------------------------------------------------------
# REVG-01: output shape equals input h shape
# ---------------------------------------------------------------------------

def test_reverse_backbone_output_shape(forward_backbone, graph_inputs):
    """REVG-01: ReverseBackbone.forward() returns same shape as input h."""
    h, edge_index = graph_inputs
    reverse = ReverseBackbone(
        forward_backbone,
        hidden_channels=16,
        separate_weights=True,
        project_embeddings=False,
    )
    out = reverse(h, edge_index)
    assert out.shape == h.shape, (
        f"Expected output shape {h.shape}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# REVG-02: weight sharing semantics
# ---------------------------------------------------------------------------

def test_separate_weights_independence(forward_backbone, graph_inputs):
    """REVG-02: separate_weights=True gives independent parameters;
    separate_weights=False registers zero own parameters."""
    h, edge_index = graph_inputs

    # (a) separate_weights=True — own parameters, no identity overlap with forward
    reverse_sep = ReverseBackbone(
        forward_backbone,
        hidden_channels=16,
        separate_weights=True,
        project_embeddings=False,
    )
    fwd_param_ids = {id(p) for p in forward_backbone.parameters()}
    rev_param_ids = {id(p) for p in reverse_sep.parameters()}
    assert len(rev_param_ids) > 0, "separate_weights=True must have own parameters"
    assert fwd_param_ids.isdisjoint(rev_param_ids), (
        "separate_weights=True must not share any parameter tensor identity "
        "with the forward backbone"
    )

    # (b) separate_weights=False — zero own backbone parameters
    reverse_shared = ReverseBackbone(
        forward_backbone,
        hidden_channels=16,
        separate_weights=False,
        project_embeddings=False,
    )
    assert len(list(reverse_shared.parameters())) == 0, (
        "separate_weights=False must register zero own parameters "
        "(shared backbone not registered as submodule)"
    )


# ---------------------------------------------------------------------------
# REVG-03: projection layer registered and correct shape
# ---------------------------------------------------------------------------

def test_project_embeddings_output_dim(forward_backbone, graph_inputs):
    """REVG-03: project_embeddings=True registers self.projection with
    in_features=2*final_dim and out_features=hidden_channels."""
    h, edge_index = graph_inputs
    hidden_channels = 16

    reverse = ReverseBackbone(
        forward_backbone,
        hidden_channels=hidden_channels,
        separate_weights=True,
        project_embeddings=True,
    )

    # Projection layer must be registered
    assert hasattr(reverse, "projection"), (
        "project_embeddings=True must register self.projection"
    )

    # Check projection dimensions
    expected_in = 2 * forward_backbone.final_dim
    assert reverse.projection.in_features == expected_in, (
        f"projection.in_features expected {expected_in}, "
        f"got {reverse.projection.in_features}"
    )
    assert reverse.projection.out_features == hidden_channels, (
        f"projection.out_features expected {hidden_channels}, "
        f"got {reverse.projection.out_features}"
    )

    # forward() still returns reverse embeddings (concatenation is Phase 5)
    out = reverse(h, edge_index)
    assert out.shape == h.shape, (
        f"forward() shape expected {h.shape}, got {out.shape}"
    )
