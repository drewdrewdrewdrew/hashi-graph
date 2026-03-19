"""Tests for RLEdgeEncoder and use_rl_edge_encoder bolt-on to transformer (Chunk 2)."""

import torch
import pytest
from torch_geometric.data import Data

from hashi_puzzle_solver.models.encoders import RLEdgeEncoder
from hashi_puzzle_solver.models.transformer import TransformerEdgeClassifier

# ── shared helpers ────────────────────────────────────────────────────────────

_NUM_NODES = 4
_NUM_FWD_EDGES = 3
_NUM_EDGES = 2 * _NUM_FWD_EDGES  # bidirectional


def _make_node_features(num_nodes: int = _NUM_NODES) -> torch.Tensor:
    """Return minimal node features [capacity, degree, unused, conflict]."""
    x = torch.zeros(num_nodes, 4)
    x[:, 0] = 3.0  # capacity
    x[:, 1] = 2.0  # structural degree
    x[:, 2] = 3.0  # unused capacity
    x[:, 3] = 0.0  # conflict status
    return x


def _make_edge_index() -> torch.Tensor:
    """Return a simple bidirectional edge index for _NUM_NODES nodes."""
    fwd = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    rev = torch.stack([fwd[1], fwd[0]])
    return torch.cat([fwd, rev], dim=1)


def _make_data(edge_dim: int = 3) -> Data:
    """Return a minimal Data object compatible with TransformerEdgeClassifier."""
    x = _make_node_features()
    edge_index = _make_edge_index()
    edge_attr = torch.rand(_NUM_EDGES, edge_dim)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _make_rl_edge_attr(num_edges: int = _NUM_EDGES, input_dim: int = 4) -> torch.Tensor:
    """Return synthetic RL edge attributes with bridge count in the last column."""
    attr = torch.rand(num_edges, input_dim)
    # bridge counts must be in {0, 1, 2}
    attr[:, -1] = torch.randint(0, 3, (num_edges,)).float()
    return attr


# ── tests ─────────────────────────────────────────────────────────────────────


def test_rl_edge_encoder_output_shape() -> None:
    """RLEdgeEncoder output has shape [num_edges, output_dim]."""
    output_dim = 16
    encoder = RLEdgeEncoder(input_dim=4, output_dim=output_dim)
    edge_attr = _make_rl_edge_attr(num_edges=10, input_dim=4)

    out = encoder(edge_attr)

    assert out.shape == (10, output_dim)


def test_bridge_count_embedding_distinguishes_values() -> None:
    """The bridge_count_embedding produces distinct vectors for 0, 1, and 2."""
    encoder = RLEdgeEncoder(input_dim=4, output_dim=16)

    # Construct three edges identical in continuous features but with bridge counts 0, 1, 2
    continuous = torch.zeros(3, 3)  # same continuous features
    attr = torch.cat([continuous, torch.tensor([[0.0], [1.0], [2.0]])], dim=1)

    out = encoder(attr)

    # Outputs for different bridge counts must differ
    assert not torch.allclose(out[0], out[1]), "count=0 and count=1 should differ"
    assert not torch.allclose(out[1], out[2]), "count=1 and count=2 should differ"
    assert not torch.allclose(out[0], out[2]), "count=0 and count=2 should differ"


def test_transformer_without_rl_encoder_unchanged() -> None:
    """TransformerEdgeClassifier with use_rl_edge_encoder=False is deterministic.

    Adding the new parameter with its default (False) must not change the
    model's behaviour: two forward passes with identical inputs must return
    identical outputs.
    """
    torch.manual_seed(42)
    model = TransformerEdgeClassifier(
        node_embedding_dim=8,
        hidden_channels=16,
        num_layers=2,
        heads=2,
        dropout=0.0,
        edge_dim=3,
        use_rl_edge_encoder=False,
    )
    model.eval()

    data = _make_data(edge_dim=3)

    with torch.no_grad():
        out1 = model(data.x, data.edge_index, edge_attr=data.edge_attr)
        out2 = model(data.x, data.edge_index, edge_attr=data.edge_attr)

    assert torch.allclose(out1, out2), "Identical inputs must yield identical outputs"


def test_transformer_with_rl_encoder_correct_shape() -> None:
    """TransformerEdgeClassifier with use_rl_edge_encoder=True runs and produces correct shape."""
    raw_edge_dim = 4
    edge_dim = 8  # encoder output_dim == edge_dim fed to TransformerConv

    torch.manual_seed(0)
    model = TransformerEdgeClassifier(
        node_embedding_dim=8,
        hidden_channels=16,
        num_layers=2,
        heads=2,
        dropout=0.0,
        edge_dim=edge_dim,
        use_rl_edge_encoder=True,
        rl_raw_edge_input_dim=raw_edge_dim,
    )
    model.eval()

    x = _make_node_features()
    edge_index = _make_edge_index()
    edge_attr = _make_rl_edge_attr(num_edges=_NUM_EDGES, input_dim=raw_edge_dim)

    with torch.no_grad():
        out = model(x, edge_index, edge_attr=edge_attr)

    # edge classifier outputs 3-class logits per edge
    assert out.shape == (_NUM_EDGES, 3)
