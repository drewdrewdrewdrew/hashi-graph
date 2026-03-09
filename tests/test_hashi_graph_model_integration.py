"""Integration tests for HashiGraphModel with IterativeBackbone + Park et al. reverse.

Covers:
  SC-1: Reasoning-only forward completes with correct shape
  SC-2: Baseline (both flags disabled) is deterministic
  SC-3: Reverse enabled — fixed-point iteration runs, output shape correct
  SC-4: EdgeHead receives correct node_hidden_dim in all valid flag combinations
  SC-5: enforce_lipschitz clamps weights correctly
  SC-6: reverse_gnn without reasoning raises ValueError
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
    fixed_point_iterations: int = 4,
    lipschitz_coeff: float = 0.99,
) -> HashiModelConfig:
    """Build a minimal HashiModelConfig for integration testing."""
    model_cfg = ModelConfig(
        type="transformer",
        hidden_channels=16,
        num_layers=2,
        heads=1,
        dropout=0.0,
        use_capacity=True,
        use_structural_degree=True,
        use_unused_capacity=True,
        use_conflict_status=False,
        use_closeness_centrality=False,
        use_articulation_points=False,
        use_spectral_features=False,
        use_global_meta_node=False,
        use_row_col_meta=False,
        use_meta_mesh=False,
        use_meta_row_col_edges=False,
        use_component_meta=False,
        use_hierarchical_component_meta=False,
        edge_concat_global_meta=False,
        edge_concat_component_meta=False,
        use_distance=False,
        use_conflict_edges=False,
        use_potential_crossing=False,
        use_categorical_edge_types=False,
        use_edge_labels_as_features=True,
        use_continuous_edge_labels=False,
        use_cut_edges=False,
        use_edge_features_in_prediction=False,
        use_noise_in_message_passing=False,
        use_noise_in_prediction=False,
        use_noise_in_global_meta=False,
        use_noise_head=False,
        use_verification_head=False,
        reasoning=ReasoningConfig(enabled=reasoning_enabled, steps=steps),
        reverse_gnn=ReverseGnnConfig(
            enabled=reverse_gnn_enabled,
            fixed_point_iterations=fixed_point_iterations,
            lipschitz_coeff=lipschitz_coeff,
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
    fixed_point_iterations: int = 4,
    lipschitz_coeff: float = 0.99,
) -> torch.nn.Module:
    """Build and return a HashiGraphModel via ModelFactory."""
    config = _make_config(
        reasoning_enabled=reasoning_enabled,
        reverse_gnn_enabled=reverse_gnn_enabled,
        steps=steps,
        fixed_point_iterations=fixed_point_iterations,
        lipschitz_coeff=lipschitz_coeff,
    )
    return ModelFactory.create_model(config, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Batch fixture
# ---------------------------------------------------------------------------

def _make_batch(n_nodes: int = 6, n_edges: int = 8):
    """Return tensors suitable for HashiGraphModel.forward()."""
    torch.manual_seed(0)
    capacity = torch.randint(0, 4, (n_nodes,))
    degree = torch.randint(0, 4, (n_nodes,))
    unused = torch.rand(n_nodes, 1)
    x = torch.cat([capacity.unsqueeze(1).float(), degree.unsqueeze(1).float(), unused], dim=1)

    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)

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
# SC-1: Reasoning only
# ---------------------------------------------------------------------------

def test_reasoning_only() -> None:
    """SC-1: reasoning.enabled=True, reverse_gnn disabled — forward completes, shape correct."""
    torch.manual_seed(0)
    model = _make_model(reasoning_enabled=True, reverse_gnn_enabled=False, steps=2)
    model.eval()

    data = _make_batch()
    with torch.no_grad():
        out = model(**data)

    n_edges = data["edge_index"].shape[1]
    assert out.shape[0] == n_edges


# ---------------------------------------------------------------------------
# SC-3: Both flags enabled — reverse fixed-point iteration
# ---------------------------------------------------------------------------

def test_both_flags_enabled() -> None:
    """SC-3: Both reasoning and reverse_gnn enabled — no error, correct shape."""
    torch.manual_seed(0)
    model = _make_model(
        reasoning_enabled=True,
        reverse_gnn_enabled=True,
        steps=2,
        fixed_point_iterations=4,
    )
    model.eval()

    data = _make_batch()
    with torch.no_grad():
        out = model(**data)

    n_edges = data["edge_index"].shape[1]
    assert out.shape[0] == n_edges


def test_reverse_creates_projection() -> None:
    """When reverse is enabled, IterativeBackbone has a projection layer."""
    model = _make_model(reasoning_enabled=True, reverse_gnn_enabled=True)
    assert hasattr(model.iterative_backbone, "projection")
    assert model.iterative_backbone.reverse_enabled is True


def test_reverse_no_projection_when_disabled() -> None:
    """When reverse is disabled, IterativeBackbone has no projection layer."""
    model = _make_model(reasoning_enabled=True, reverse_gnn_enabled=False)
    assert not hasattr(model.iterative_backbone, "projection")
    assert model.iterative_backbone.reverse_enabled is False


# ---------------------------------------------------------------------------
# SC-5: enforce_lipschitz clamps weights
# ---------------------------------------------------------------------------

def test_enforce_lipschitz() -> None:
    """enforce_lipschitz clamps conv weight Frobenius norms to <= c."""
    model = _make_model(reasoning_enabled=True, reverse_gnn_enabled=True)
    bb = model.iterative_backbone
    c = 0.5

    # Inflate a weight so it clearly exceeds c
    with torch.no_grad():
        bb.conv.lin_key.weight.mul_(100.0)

    bb.enforce_lipschitz(c)

    norm_after = bb.conv.lin_key.weight.norm(p="fro").item()
    assert norm_after <= c + 1e-5, f"Expected ||W||_F <= {c}, got {norm_after}"


# ---------------------------------------------------------------------------
# SC-6: reverse_gnn without reasoning raises ValueError
# ---------------------------------------------------------------------------

def test_reverse_without_reasoning_raises() -> None:
    """reverse_gnn.enabled=True without reasoning.enabled=True should raise."""
    with pytest.raises(ValueError, match="reverse_gnn requires reasoning"):
        _make_model(reasoning_enabled=False, reverse_gnn_enabled=True)


# ---------------------------------------------------------------------------
# SC-4: EdgeHead receives correct dim in valid flag combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("reasoning_enabled", "reverse_gnn_enabled"),
    [
        (False, False),   # baseline
        (True,  False),   # reasoning only
        (True,  True),    # both — projection maps back to hidden_channels
    ],
)
def test_edge_head_dim_all_combos(
    reasoning_enabled: bool,
    reverse_gnn_enabled: bool,
) -> None:
    """SC-4: No shape mismatch RuntimeError for any valid flag combination."""
    torch.manual_seed(0)
    model = _make_model(
        reasoning_enabled=reasoning_enabled,
        reverse_gnn_enabled=reverse_gnn_enabled,
    )
    model.eval()

    data = _make_batch()
    with torch.no_grad():
        out = model(**data)

    n_edges = data["edge_index"].shape[1]
    assert out.shape[0] == n_edges
