"""Tests for IterativeBackbone — REAS-01 and REAS-02.

Tests are written against the target contract before implementation exists
(TDD RED phase). They will fail with ModuleNotFoundError until Task 1 completes.
"""

import torch
import torch.nn.functional as func

from hashi_puzzle_solver.models.iterative_backbone import IterativeBackbone


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_graph(hidden_channels: int = 16, edge_dim: int = 4):
    """Return a minimal synthetic graph: 4 nodes, 6 edges."""
    torch.manual_seed(42)
    n_nodes = 4
    h = torch.randn(n_nodes, hidden_channels)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3],
         [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    edge_attr = torch.randn(edge_index.shape[1], edge_dim)
    return h, edge_index, edge_attr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_iterative_backbone_applies_k_times():
    """REAS-01: shared conv is called exactly steps times per forward pass."""
    torch.manual_seed(42)
    hidden_channels = 16
    edge_dim = 4
    steps = 3

    backbone = IterativeBackbone(
        hidden_channels=hidden_channels,
        steps=steps,
        heads=1,
        dropout=0.0,
        edge_dim=edge_dim,
    )
    h, edge_index, edge_attr = _make_graph(hidden_channels, edge_dim)

    # Spy on conv.forward by replacing it with a counting wrapper.
    # We cannot use patch.object on a nn.Module attribute (PyTorch rejects
    # non-Module assignments to registered submodules), so we wrap the method.
    call_count = 0
    _original_forward = backbone.conv.forward

    def spy_forward(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _original_forward(*args, **kwargs)

    backbone.conv.forward = spy_forward  # type: ignore[method-assign]

    backbone.eval()
    with torch.no_grad():
        _ = backbone(h, edge_index, edge_attr=edge_attr)

    assert call_count == steps, (
        f"Expected conv to be called {steps} times, got {call_count}"
    )


def test_iterative_backbone_steps_parameter():
    """REAS-02: steps attribute matches constructor argument; output shape matches input."""
    torch.manual_seed(42)
    hidden_channels = 16
    edge_dim = 4
    steps = 5

    backbone = IterativeBackbone(
        hidden_channels=hidden_channels,
        steps=steps,
        heads=1,
        dropout=0.0,
        edge_dim=edge_dim,
    )
    h, edge_index, edge_attr = _make_graph(hidden_channels, edge_dim)

    assert backbone.steps == steps, (
        f"backbone.steps should be {steps}, got {backbone.steps}"
    )

    backbone.eval()
    with torch.no_grad():
        out = backbone(h, edge_index, edge_attr=edge_attr)

    assert out.shape == h.shape, (
        f"Output shape {out.shape} does not match input shape {h.shape}"
    )


def test_steps_one_matches_single_pass():
    """steps=1 output is identical to one manual conv->norm->relu->residual pass."""
    torch.manual_seed(42)
    hidden_channels = 16
    edge_dim = 4

    backbone = IterativeBackbone(
        hidden_channels=hidden_channels,
        steps=1,
        heads=1,
        dropout=0.25,  # will be zeroed by eval()
        edge_dim=edge_dim,
    )
    backbone.eval()

    h, edge_index, edge_attr = _make_graph(hidden_channels, edge_dim)

    # Run backbone forward
    with torch.no_grad():
        result = backbone(h, edge_index, edge_attr=edge_attr)

    # Manually replicate: conv -> norm -> relu -> dropout(p=0, eval) -> residual
    with torch.no_grad():
        h_in = h
        h_manual = backbone.conv(h, edge_index, edge_attr=edge_attr)
        h_manual = backbone.norm(h_manual)
        h_manual = func.relu(h_manual)
        h_manual = func.dropout(h_manual, p=0.0, training=False)
        expected = h_manual + h_in

    assert torch.allclose(result, expected, atol=1e-6), (
        f"steps=1 result differs from manual single pass. "
        f"Max diff: {(result - expected).abs().max().item()}"
    )
