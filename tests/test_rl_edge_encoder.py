"""Tests for RLEdgeEncoder, SchemaRLEdgeEncoder, and RL encoder wiring (Part 1)."""

import torch
import pytest
from torch_geometric.data import Data

from hashi_puzzle_solver.models.config import ModelConfig
from hashi_puzzle_solver.models.encoders import RLEdgeEncoder, SchemaRLEdgeEncoder
from hashi_puzzle_solver.models.features import EdgeFeatureManager, edge_label_column_indices
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


def test_transformer_with_rl_encoder_and_categorical_edge_types() -> None:
    """TransformerEdgeClassifier with both RL encoder and categorical edge types works correctly.
    
    This test catches the bug where categorical edge type embeddings are concatenated
    before the RL encoder, causing dimension mismatches. The RL encoder should process
    raw features first, then categorical edge types are added.
    """
    raw_edge_dim = 71  # Realistic dimension from full feature set + bridge count
    edge_dim = 16  # RL encoder output dimension
    node_embedding_dim = 64
    
    torch.manual_seed(0)
    model = TransformerEdgeClassifier(
        node_embedding_dim=node_embedding_dim,
        hidden_channels=256,
        num_layers=4,
        heads=8,
        dropout=0.25,
        edge_dim=edge_dim,
        use_rl_edge_encoder=True,
        rl_raw_edge_input_dim=raw_edge_dim,
        use_categorical_edge_types=True,
    )
    model.eval()
    
    # Create realistic node features (capacity, degree, unused, conflict)
    num_nodes = 10
    x = torch.zeros(num_nodes, 4)
    x[:, 0] = torch.randint(1, 8, (num_nodes,)).float()  # capacity
    x[:, 1] = torch.randint(1, 5, (num_nodes,)).float()  # structural degree
    x[:, 2] = torch.randint(0, 8, (num_nodes,)).float()  # unused capacity
    x[:, 3] = torch.randint(0, 2, (num_nodes,)).float()  # conflict status
    
    # Create edge index
    num_edges = 20
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create edge attributes with full feature dimension
    edge_attr = _make_rl_edge_attr(num_edges=num_edges, input_dim=raw_edge_dim)
    
    # Create edge types (puzzle edges = 0)
    edge_type = torch.zeros(num_edges, dtype=torch.long)
    
    with torch.no_grad():
        out = model(x, edge_index, edge_attr=edge_attr, edge_type=edge_type)
    
    # Should output 3-class logits per edge
    assert out.shape == (num_edges, 3), f"Expected shape ({num_edges}, 3), got {out.shape}"


# ── schema helper tests ───────────────────────────────────────────────────────


def _make_model_config(**overrides: object) -> ModelConfig:
    """Return a minimal ModelConfig with edge labels enabled."""
    defaults = dict(
        use_edge_labels_as_features=True,
        use_categorical_edge_types=False,
        use_conflict_edges=False,
        use_meta_mesh=False,
        use_meta_row_col_edges=False,
        use_cut_edges=False,
        use_potential_crossing=False,
        use_continuous_edge_labels=False,
        use_component_meta=False,
        use_boundary_flag=False,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_edge_label_column_indices_returns_none_when_disabled() -> None:
    """edge_label_column_indices returns None when the feature flag is off."""
    mc = ModelConfig(use_edge_labels_as_features=False)
    assert edge_label_column_indices(mc) is None


def test_edge_label_column_indices_non_categorical() -> None:
    """Indices are correct for the non-categorical layout (is_meta present)."""
    mc = _make_model_config(use_categorical_edge_types=False)
    fm = EdgeFeatureManager(mc)
    result = edge_label_column_indices(mc)
    assert result is not None
    bridge_idx, labeled_idx = result
    assert bridge_idx == fm.get_idx("bridge_label")
    assert labeled_idx == fm.get_idx("is_labeled")
    assert labeled_idx == bridge_idx + 1


def test_edge_label_column_indices_categorical_no_is_meta() -> None:
    """When use_categorical_edge_types=True, is_meta is absent so bridge_label
    appears two columns earlier than the legacy hand-rolled calculation assumed.

    Regression test for the masking.py drift bug.
    """
    mc_cat = _make_model_config(use_categorical_edge_types=True)
    mc_non = _make_model_config(use_categorical_edge_types=False)

    cat_bridge, _ = edge_label_column_indices(mc_cat)
    non_bridge, _ = edge_label_column_indices(mc_non)

    # Categorical layout omits is_meta (1 col) so bridge_label is one step earlier
    assert cat_bridge == non_bridge - 1, (
        f"Expected categorical bridge_label at {non_bridge - 1}, got {cat_bridge}"
    )


def test_edge_label_column_indices_with_optional_flags() -> None:
    """Adding optional flags shifts both indices by the same amount."""
    base = _make_model_config(use_categorical_edge_types=False)
    with_conflict = _make_model_config(
        use_categorical_edge_types=False,
        use_conflict_edges=True,
    )
    base_b, base_l = edge_label_column_indices(base)
    conf_b, conf_l = edge_label_column_indices(with_conflict)

    assert conf_b == base_b + 1
    assert conf_l == base_l + 1


# ── SchemaRLEdgeEncoder tests ─────────────────────────────────────────────────


def _make_schema_feature_manager(**overrides: object) -> EdgeFeatureManager:
    return EdgeFeatureManager(_make_model_config(**overrides))


def _make_schema_edge_attr(fm: EdgeFeatureManager, num_edges: int = 6) -> torch.Tensor:
    """Synthetic edge_attr matching the feature manager schema."""
    attr = torch.rand(num_edges, fm.num_edge_feats)
    bl_idx = fm.get_idx("bridge_label")
    il_idx = fm.get_idx("is_labeled")
    attr[:, bl_idx] = torch.randint(0, 3, (num_edges,)).float()
    attr[:, il_idx] = torch.randint(0, 2, (num_edges,)).float()
    return attr


def test_schema_rl_encoder_output_shape() -> None:
    """SchemaRLEdgeEncoder output has shape [num_edges, output_dim]."""
    fm = _make_schema_feature_manager()
    output_dim = 16
    encoder = SchemaRLEdgeEncoder(feature_manager=fm, output_dim=output_dim)

    edge_attr = _make_schema_edge_attr(fm, num_edges=10)
    out = encoder(edge_attr)

    assert out.shape == (10, output_dim)


def test_schema_rl_encoder_requires_labels_feature() -> None:
    """SchemaRLEdgeEncoder raises when use_edge_labels_as_features is off."""
    mc = ModelConfig(use_edge_labels_as_features=False)
    fm = EdgeFeatureManager(mc)
    with pytest.raises(ValueError, match="use_edge_labels_as_features"):
        SchemaRLEdgeEncoder(feature_manager=fm, output_dim=16)


def test_schema_rl_encoder_bridge_label_embeddings_distinct() -> None:
    """bridge_label_embedding produces distinct vectors for 0, 1, 2."""
    fm = _make_schema_feature_manager()
    encoder = SchemaRLEdgeEncoder(feature_manager=fm, output_dim=16)

    bl_idx = fm.get_idx("bridge_label")
    total = fm.num_edge_feats

    base = torch.zeros(3, total)
    base[:, bl_idx] = torch.tensor([0.0, 1.0, 2.0])

    out = encoder(base)

    assert not torch.allclose(out[0], out[1]), "bridge_label 0 and 1 must differ"
    assert not torch.allclose(out[1], out[2]), "bridge_label 1 and 2 must differ"
    assert not torch.allclose(out[0], out[2]), "bridge_label 0 and 2 must differ"


def test_schema_rl_encoder_is_labeled_embeddings_distinct() -> None:
    """is_labeled_embedding produces distinct vectors for 0 and 1."""
    fm = _make_schema_feature_manager()
    encoder = SchemaRLEdgeEncoder(feature_manager=fm, output_dim=16)

    il_idx = fm.get_idx("is_labeled")
    total = fm.num_edge_feats

    base = torch.zeros(2, total)
    base[:, il_idx] = torch.tensor([0.0, 1.0])

    out = encoder(base)
    assert not torch.allclose(out[0], out[1]), "is_labeled 0 and 1 must differ"


def test_schema_rl_encoder_gradient_flow() -> None:
    """Gradients flow back through SchemaRLEdgeEncoder."""
    fm = _make_schema_feature_manager()
    encoder = SchemaRLEdgeEncoder(feature_manager=fm, output_dim=16)

    edge_attr = _make_schema_edge_attr(fm, num_edges=8)
    out = encoder(edge_attr)
    loss = out.sum()
    loss.backward()

    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_schema_rl_encoder_categorical_layout() -> None:
    """SchemaRLEdgeEncoder works correctly when use_categorical_edge_types=True."""
    fm = _make_schema_feature_manager(use_categorical_edge_types=True)
    output_dim = 8
    encoder = SchemaRLEdgeEncoder(feature_manager=fm, output_dim=output_dim)

    edge_attr = _make_schema_edge_attr(fm, num_edges=5)
    out = encoder(edge_attr)
    assert out.shape == (5, output_dim)


def test_schema_rl_encoder_with_all_optional_flags() -> None:
    """SchemaRLEdgeEncoder handles a large feature set without index errors."""
    fm = _make_schema_feature_manager(
        use_conflict_edges=True,
        use_meta_mesh=True,
        use_meta_row_col_edges=True,
        use_cut_edges=True,
        use_potential_crossing=True,
        use_continuous_edge_labels=True,
    )
    encoder = SchemaRLEdgeEncoder(feature_manager=fm, output_dim=32)
    edge_attr = _make_schema_edge_attr(fm, num_edges=12)
    out = encoder(edge_attr)
    assert out.shape == (12, 32)


# ── TransformerEdgeClassifier + SchemaRLEdgeEncoder wiring ───────────────────


def test_transformer_with_schema_encoder_correct_shape() -> None:
    """TransformerEdgeClassifier with rl_edge_feature_manager uses SchemaRLEdgeEncoder."""
    fm = _make_schema_feature_manager()
    edge_dim = 8

    torch.manual_seed(1)
    model = TransformerEdgeClassifier(
        node_embedding_dim=8,
        hidden_channels=16,
        num_layers=2,
        heads=2,
        dropout=0.0,
        edge_dim=edge_dim,
        use_rl_edge_encoder=True,
        rl_edge_feature_manager=fm,
    )
    assert isinstance(model.rl_edge_encoder, SchemaRLEdgeEncoder)

    x = _make_node_features()
    edge_index = _make_edge_index()
    edge_attr = _make_schema_edge_attr(fm, num_edges=_NUM_EDGES)

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_attr=edge_attr)

    assert out.shape == (_NUM_EDGES, 3)


def test_transformer_without_feature_manager_uses_legacy_encoder() -> None:
    """Without rl_edge_feature_manager, legacy RLEdgeEncoder is used for back-compat."""
    torch.manual_seed(2)
    model = TransformerEdgeClassifier(
        node_embedding_dim=8,
        hidden_channels=16,
        num_layers=2,
        heads=2,
        dropout=0.0,
        edge_dim=8,
        use_rl_edge_encoder=True,
        rl_raw_edge_input_dim=4,
    )
    assert isinstance(model.rl_edge_encoder, RLEdgeEncoder)


def test_transformer_schema_encoder_gradient_flow() -> None:
    """Gradients flow end-to-end through the full model with SchemaRLEdgeEncoder."""
    fm = _make_schema_feature_manager()
    edge_dim = 8

    torch.manual_seed(3)
    model = TransformerEdgeClassifier(
        node_embedding_dim=8,
        hidden_channels=16,
        num_layers=2,
        heads=2,
        dropout=0.0,
        edge_dim=edge_dim,
        use_rl_edge_encoder=True,
        rl_edge_feature_manager=fm,
    )

    x = _make_node_features()
    edge_index = _make_edge_index()
    edge_attr = _make_schema_edge_attr(fm, num_edges=_NUM_EDGES)

    out = model(x, edge_index, edge_attr=edge_attr)
    loss = out.sum()
    loss.backward()

    for name, param in model.rl_edge_encoder.named_parameters():
        assert param.grad is not None, f"No gradient for encoder.{name}"


# ── masking regression tests ──────────────────────────────────────────────────


def test_rl_loader_integration_with_full_config() -> None:
    """Test that the RL loader and model builder work with a realistic config.
    
    This simulates the full pipeline: dataset -> loader -> model -> forward pass.
    """
    from hashi_puzzle_solver.rl.loader import build_rl_model
    
    # Simulate a config with many features enabled (like rl_sequential.yaml)
    config = {
        "model": {
            "type": "transformer",
            "node_embedding_dim": 64,
            "hidden_channels": 256,
            "num_layers": 4,
            "heads": 8,
            "dropout": 0.25,
            "use_categorical_edge_types": True,
            "use_continuous_edge_labels": True,
            "use_distance": True,
            "use_conflict_edges": True,
            "use_potential_crossing": True,
            "use_cut_edges": True,
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_conflict_status": True,
            "use_closeness_centrality": True,
            "use_articulation_points": True,
            "use_spectral_features": True,
            "use_global_meta_node": True,
            "use_row_col_meta": True,
            "use_meta_mesh": True,
            "use_meta_row_col_edges": True,
            "edge_type_embedding_dim": 8,
            "logit_embedding_dim": 16,
        }
    }
    
    # Simulate edge_attr dimension from dataset with all features enabled
    # This would be calculated from EdgeFeatureManager + 1 bridge count column
    # Approximate: inv_dx(1) + inv_dy(1) + is_meta(1) + is_conflict(1) + 
    #              is_meta_mesh(1) + is_meta_row_col_cross(1) + is_cut_edge(1) + 
    #              is_potential_crossing(1) + bridge_logits(3) + bridge_count(1) = 12
    # But with more features in practice it can be much larger (e.g., 71)
    edge_attr_dim = 71
    
    device = torch.device("cpu")
    model = build_rl_model(config, edge_attr_dim, device)
    model.eval()
    
    # Create test data with proper node feature columns
    # Node features: capacity, degree, unused, conflict, closeness, ap, spectral(3)
    num_nodes = 10
    num_edges = 20
    x = torch.zeros(num_nodes, 9)  # 9 columns for all enabled features
    x[:, 0] = torch.randint(1, 8, (num_nodes,)).float()  # capacity
    x[:, 1] = torch.randint(1, 5, (num_nodes,)).float()  # structural degree
    x[:, 2] = torch.randint(0, 8, (num_nodes,)).float()  # unused capacity
    x[:, 3] = torch.randint(0, 2, (num_nodes,)).float()  # conflict status
    x[:, 4] = torch.rand(num_nodes)  # closeness centrality
    x[:, 5] = torch.randint(0, 2, (num_nodes,)).float()  # articulation points
    x[:, 6:9] = torch.randn(num_nodes, 3)  # spectral features
    
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = _make_rl_edge_attr(num_edges=num_edges, input_dim=edge_attr_dim)
    edge_type = torch.zeros(num_edges, dtype=torch.long)
    
    with torch.no_grad():
        out = model(x, edge_index, edge_attr=edge_attr, edge_type=edge_type)
    
    assert out.shape == (num_edges, 3), f"Expected shape ({num_edges}, 3), got {out.shape}"


# ── masking regression: bridge_label index with categorical edge types ────────


def test_masking_bridge_label_index_categorical_vs_non_categorical() -> None:
    """Regression: bridge_label index shifts correctly when is_meta is absent.

    With use_categorical_edge_types=True the EdgeFeatureManager omits is_meta,
    so bridge_label appears one column earlier.  The old hand-rolled code in
    masking.py always assumed is_meta was present (current_idx = 3) which
    produced the wrong column for categorical configs.
    """
    mc_cat = ModelConfig(
        use_edge_labels_as_features=True,
        use_categorical_edge_types=True,
        use_conflict_edges=False,
        use_meta_mesh=False,
        use_meta_row_col_edges=False,
        use_cut_edges=False,
        use_potential_crossing=False,
        use_continuous_edge_labels=False,
    )
    mc_non = ModelConfig(
        use_edge_labels_as_features=True,
        use_categorical_edge_types=False,
        use_conflict_edges=False,
        use_meta_mesh=False,
        use_meta_row_col_edges=False,
        use_cut_edges=False,
        use_potential_crossing=False,
        use_continuous_edge_labels=False,
    )

    fm_cat = EdgeFeatureManager(mc_cat)
    fm_non = EdgeFeatureManager(mc_non)

    cat_bl = fm_cat.get_idx("bridge_label")
    non_bl = fm_non.get_idx("bridge_label")

    # Non-categorical has: inv_dx(0), inv_dy(1), is_meta(2), bridge_label(3)
    assert non_bl == 3, f"Expected non-categorical bridge_label at col 3, got {non_bl}"
    # Categorical has: inv_dx(0), inv_dy(1), bridge_label(2)
    assert cat_bl == 2, f"Expected categorical bridge_label at col 2, got {cat_bl}"

    # edge_label_column_indices must match EdgeFeatureManager directly
    assert edge_label_column_indices(mc_cat) == (cat_bl, cat_bl + 1)
    assert edge_label_column_indices(mc_non) == (non_bl, non_bl + 1)
