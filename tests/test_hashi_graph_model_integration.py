"""Integration tests for HashiGraphModel with optional IterativeBackbone and ReverseBackbone.

TDD Wave 0 (RED): Tests fail before Task 1 (core.py + factory.py changes) because
HashiGraphModel does not yet accept iterative_backbone/reverse_backbone args and
the interleaved composition block does not exist.

Covers Phase 5 success criteria:
  SC-1: Iterative + Reverse composition runs end-to-end
  SC-2: Baseline (both flags disabled) is byte-for-byte identical
  SC-3: With both enabled, reverse_backbone called K times (once per reasoning step)
  SC-4: EdgeHead receives correct node_hidden_dim in all four flag combinations
"""

from __future__ import annotations

import torch
import pytest

from hashi_puzzle_solver.models.config import (
    HashiModelConfig,
    ModelConfig,
    ReasoningConfig,
    ReverseGnnConfig,
    DataConfig,
    TrainingConfig,
)
from hashi_puzzle_solver.models.factory import ModelFactory


# ---------------------------------------------------------------------------
# Config and model factory helpers
# ---------------------------------------------------------------------------

def _make_config(
    reasoning_enabled: bool = False,
    reverse_gnn_enabled: bool = False,
    steps: int = 2,
    project_embeddings: bool = True,
    separate_weights: bool = True,
) -> HashiModelConfig:
    """Build a minimal HashiModelConfig for integration testing.

    Uses hidden_channels=16 to keep test tensors small. All complex features
    are disabled to minimise node feature dimension.
    """
    model_cfg = ModelConfig(
        type="transformer",
        hidden_channels=16,
        num_layers=2,
        heads=1,
        dropout=0.0,
        # Disable spectral / structural extras so node encoder stays small
        use_capacity=True,
        use_structural_degree=True,
        use_unused_capacity=True,
        use_conflict_status=False,
        use_closeness_centrality=False,
        use_articulation_points=False,
        use_spectral_features=False,
        # Keep meta nodes but disable their edges to reduce edge encoder dim
        use_global_meta_node=False,
        use_row_col_meta=False,
        use_meta_mesh=False,
        use_meta_row_col_edges=False,
        use_component_meta=False,
        use_hierarchical_component_meta=False,
        # Disable global meta concat in EdgeHead (no meta nodes in test graph)
        edge_concat_global_meta=False,
        edge_concat_component_meta=False,
        # Edge features — minimal set
        use_distance=False,
        use_conflict_edges=False,
        use_potential_crossing=False,
        use_categorical_edge_types=False,
        use_edge_labels_as_features=True,
        use_continuous_edge_labels=False,
        use_cut_edges=False,
        use_edge_features_in_prediction=False,
        # Noise — off
        use_noise_in_message_passing=False,
        use_noise_in_prediction=False,
        use_noise_in_global_meta=False,
        use_noise_head=False,
        # Heads
        use_verification_head=False,
        # Reasoning / Reverse GNN flags under test
        reasoning=ReasoningConfig(enabled=reasoning_enabled, steps=steps),
        reverse_gnn=ReverseGnnConfig(
            enabled=reverse_gnn_enabled,
            separate_weights=separate_weights,
            project_embeddings=project_embeddings,
        ),
    )
    return HashiModelConfig(
        data=DataConfig(),
        model=model_cfg,
        training=TrainingConfig(),
    )


def _make_model(
    reasoning_enabled: bool = False,
    reverse_gnn_enabled: bool = False,
    steps: int = 2,
    project_embeddings: bool = True,
    separate_weights: bool = True,
) -> torch.nn.Module:
    """Build and return a HashiGraphModel via ModelFactory."""
    config = _make_config(
        reasoning_enabled=reasoning_enabled,
        reverse_gnn_enabled=reverse_gnn_enabled,
        steps=steps,
        project_embeddings=project_embeddings,
        separate_weights=separate_weights,
    )
    return ModelFactory.create_model(config, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Batch fixture
# ---------------------------------------------------------------------------

def _make_batch(n_nodes: int = 6, n_edges: int = 8):
    """Return tensors suitable for HashiGraphModel.forward().

    Node features: capacity(int), structural_degree(int), unused_capacity(float)
    Edge features must match the EdgeFeatureManager schema for the test config
    (no categorical edge types, use_edge_labels_as_features=True):
      col 0: inv_dx
      col 1: inv_dy
      col 2: is_meta (present because use_categorical_edge_types=False)
      col 3: bridge_label (from use_edge_labels_as_features)
      col 4: is_labeled  (from use_edge_labels_as_features)
    Total: 5 columns.
    """
    torch.manual_seed(0)
    # capacity [0..3], structural_degree [0..3]
    capacity = torch.randint(0, 4, (n_nodes,))
    degree = torch.randint(0, 4, (n_nodes,))
    unused = torch.rand(n_nodes, 1)
    x = torch.cat([capacity.unsqueeze(1).float(), degree.unsqueeze(1).float(), unused], dim=1)

    # Random sparse edge_index with n_edges edges
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)

    # Edge attr: 5 columns matching the EdgeFeatureManager schema described above
    edge_attr = torch.randn(n_edges, 5)

    batch = torch.zeros(n_nodes, dtype=torch.long)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch,
    }


# ---------------------------------------------------------------------------
# SC-2: Baseline — both flags disabled
# ---------------------------------------------------------------------------

def test_flags_disabled_baseline() -> None:
    """SC-2: With both flags disabled, two forward passes produce identical output."""
    torch.manual_seed(0)
    model = _make_model(reasoning_enabled=False, reverse_gnn_enabled=False)
    model.eval()

    data = _make_batch()

    with torch.no_grad():
        torch.manual_seed(0)
        out1 = model(**data)
        torch.manual_seed(0)
        out2 = model(**data)

    assert torch.equal(out1, out2), (
        "Baseline (both flags disabled) must produce identical output across two runs in eval mode"
    )


# ---------------------------------------------------------------------------
# SC-1 partial: Reasoning only
# ---------------------------------------------------------------------------

def test_reasoning_only() -> None:
    """SC-1 partial: reasoning.enabled=True, reverse_gnn disabled — forward completes, shape correct."""
    torch.manual_seed(0)
    model = _make_model(reasoning_enabled=True, reverse_gnn_enabled=False, steps=2)
    model.eval()

    data = _make_batch()
    with torch.no_grad():
        out = model(**data)

    n_edges = data["edge_index"].shape[1]
    assert out.shape[0] == n_edges, (
        f"Output first dim should be n_edges={n_edges}, got {out.shape[0]}"
    )


# ---------------------------------------------------------------------------
# SC-4 partial: Reverse only with projection
# ---------------------------------------------------------------------------

def test_reverse_only_with_projection() -> None:
    """SC-4: reverse_gnn enabled, project_embeddings=True — no RuntimeError."""
    torch.manual_seed(0)
    model = _make_model(
        reasoning_enabled=False,
        reverse_gnn_enabled=True,
        project_embeddings=True,
    )
    model.eval()

    data = _make_batch()
    with torch.no_grad():
        out = model(**data)

    n_edges = data["edge_index"].shape[1]
    assert out.shape[0] == n_edges


def test_reverse_only_no_projection() -> None:
    """SC-4: reverse_gnn enabled, project_embeddings=False (2*final_dim path) — no RuntimeError."""
    torch.manual_seed(0)
    model = _make_model(
        reasoning_enabled=False,
        reverse_gnn_enabled=True,
        project_embeddings=False,
    )
    model.eval()

    data = _make_batch()
    with torch.no_grad():
        out = model(**data)

    n_edges = data["edge_index"].shape[1]
    assert out.shape[0] == n_edges


# ---------------------------------------------------------------------------
# SC-1: Both flags enabled
# ---------------------------------------------------------------------------

def test_both_flags_enabled() -> None:
    """SC-1: Both reasoning and reverse_gnn enabled (project_embeddings=True) — no error, correct shape."""
    torch.manual_seed(0)
    model = _make_model(
        reasoning_enabled=True,
        reverse_gnn_enabled=True,
        steps=2,
        project_embeddings=True,
    )
    model.eval()

    data = _make_batch()
    with torch.no_grad():
        out = model(**data)

    n_edges = data["edge_index"].shape[1]
    assert out.shape[0] == n_edges


# ---------------------------------------------------------------------------
# SC-3: Interleaved — reverse_backbone called exactly K times
# ---------------------------------------------------------------------------

def test_rev_reasoning_interleaved() -> None:
    """SC-3: With both flags enabled and steps=3, reverse_backbone.forward called exactly 3 times."""
    torch.manual_seed(0)
    steps = 3
    model = _make_model(
        reasoning_enabled=True,
        reverse_gnn_enabled=True,
        steps=steps,
        project_embeddings=True,
    )
    model.eval()

    # Wrap reverse_backbone.forward with a call-counting spy.
    # We reassign the bound method (same pattern as test_iterative_backbone.py).
    spy_call_count = 0
    _original_forward = model.reverse_backbone.forward

    def spy_forward(*args, **kwargs):
        nonlocal spy_call_count
        spy_call_count += 1
        return _original_forward(*args, **kwargs)

    model.reverse_backbone.forward = spy_forward  # type: ignore[method-assign]

    data = _make_batch()
    with torch.no_grad():
        _ = model(**data)

    assert spy_call_count == steps, (
        f"Expected reverse_backbone.forward to be called {steps} times in interleaved loop, "
        f"got {spy_call_count}"
    )


# ---------------------------------------------------------------------------
# SC-4: EdgeHead receives correct dim in all flag combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("reasoning_enabled", "reverse_gnn_enabled", "project_embeddings"),
    [
        (False, False, True),   # baseline — backbone.final_dim
        (True,  False, True),   # reasoning only — backbone.final_dim (unchanged)
        (False, True,  True),   # reverse + projection — hidden_channels
        (False, True,  False),  # reverse no projection — 2 * final_dim
        (True,  True,  True),   # both — hidden_channels (via projection)
    ],
)
def test_edge_head_dim_all_combos(
    reasoning_enabled: bool,
    reverse_gnn_enabled: bool,
    project_embeddings: bool,
) -> None:
    """SC-4: No shape mismatch RuntimeError for any flag combination."""
    torch.manual_seed(0)
    model = _make_model(
        reasoning_enabled=reasoning_enabled,
        reverse_gnn_enabled=reverse_gnn_enabled,
        project_embeddings=project_embeddings,
    )
    model.eval()

    data = _make_batch()
    with torch.no_grad():
        # Must not raise
        out = model(**data)

    n_edges = data["edge_index"].shape[1]
    assert out.shape[0] == n_edges
