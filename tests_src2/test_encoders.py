"""Tests for modular encoders."""

import torch
from hashi_puzzle_solver.models.config import ModelConfig
from hashi_puzzle_solver.models.features import NodeFeatureManager, EdgeFeatureManager
from hashi_puzzle_solver.models.encoders import NodeEncoder, EdgeEncoder


def test_node_encoder_forward():
    """Test that NodeEncoder forward pass works and has correct shape."""
    config = ModelConfig(
        node_embedding_dim=16,
        hidden_channels=32,
        use_capacity=True,
        use_structural_degree=True,
        use_unused_capacity=True,
        use_conflict_status=False,
        use_closeness_centrality=False,
        use_articulation_points=False,
        use_spectral_features=False,
    )
    fm = NodeFeatureManager(config)
    encoder = NodeEncoder(config, fm)

    # Mock input tensor [batch_size, num_features]
    # capacity, structural_degree, unused_capacity
    x = torch.zeros((5, 3))
    x[:, 0] = torch.tensor([1, 2, 3, 4, 5])  # capacities
    x[:, 1] = torch.tensor([1, 2, 2, 3, 4])  # degrees
    x[:, 2] = torch.tensor([0.5, 1.0, 0.0, -0.5, 2.0])  # unused

    h = encoder(x)
    assert h.shape == (5, 32)


def test_edge_encoder_forward():
    """Test that EdgeEncoder forward pass works and has correct shape."""
    config = ModelConfig(
        edge_type_embedding_dim=8,
        distance_embedding_dim=16,
        use_categorical_edge_types=True,
        use_continuous_edge_labels=False, # Disable logits to match mock edge_attr
    )
    fm = EdgeFeatureManager(config)
    encoder = EdgeEncoder(config, fm)

    # Mock edge_attr [num_edges, num_edge_feats]
    # In this config, only inv_dx, inv_dy are enabled (2 feats)
    edge_attr = torch.zeros((10, 2))
    edge_type = torch.randint(0, 9, (10,))

    h_edge = encoder(edge_attr, edge_type)
    
    # Expected shape: [10, edge_type_embedding_dim (8) + distance_embedding_dim (16)]
    assert h_edge.shape == (10, 8 + 16)


def test_encoders_zero_out_disabled():
    """Test that disabling features actually removes them from the encoding process."""
    # Config 1: All features enabled (but we'll be explicit for this test)
    config_all = ModelConfig(
        capacity_embedding_dim=16,
        unused_embedding_dim=32,
        use_capacity=True, 
        use_unused_capacity=True,
        use_structural_degree=False,
        use_conflict_status=False,
        use_closeness_centrality=False,
        use_articulation_points=False,
        use_spectral_features=False,
    )
    fm_all = NodeFeatureManager(config_all)
    encoder_all = NodeEncoder(config_all, fm_all)
    
    # Config 2: Some features disabled
    config_some = ModelConfig(
        capacity_embedding_dim=16,
        use_capacity=True, 
        use_unused_capacity=False,
        use_structural_degree=False,
        use_conflict_status=False,
        use_closeness_centrality=False,
        use_articulation_points=False,
        use_spectral_features=False,
    )
    fm_some = NodeFeatureManager(config_some)
    encoder_some = NodeEncoder(config_some, fm_some)

    # Different number of input features
    x_all = torch.zeros((1, 2))
    x_some = torch.zeros((1, 1))
    
    # Encoders should be initialized and forward pass should work
    assert encoder_all(x_all).shape[0] == 1
    assert encoder_some(x_some).shape[0] == 1
    
    # Internal input dims of refiners should differ
    assert encoder_all.refiner[0].in_features == 16 + 32
    assert encoder_some.refiner[0].in_features == 16


def test_constraint_vocab_forward():
    """Constraint vocab replaces capacity/degree/unused with joint embedding."""
    config = ModelConfig(
        node_embedding_dim=16,
        hidden_channels=32,
        constraint_vocab_dim=24,
        use_constraint_vocab=True,
        use_capacity=False,
        use_structural_degree=False,
        use_structural_degree_nsew=False,
        use_unused_capacity=False,
        use_conflict_status=False,
        use_closeness_centrality=False,
        use_articulation_points=False,
        use_spectral_features=False,
    )
    fm = NodeFeatureManager(config)
    encoder = NodeEncoder(config, fm)

    # Feature tensor: [structural_degree, unused_capacity] (2 cols registered by fm)
    x = torch.tensor([
        [1.0, 0.0],   # degree=1, net_cap=0 -> satisfied
        [2.0, 3.0],   # degree=2, net_cap=3 -> partial
        [4.0, -2.0],  # degree=4, net_cap=-2 -> over-saturated
        [3.0, 8.0],   # degree=3, net_cap=8 -> max range
        [1.0, 1.0],   # degree=1, net_cap=1 -> forced
    ])

    h = encoder(x)
    assert h.shape == (5, 32)


def test_constraint_vocab_index_bounds():
    """Constraint vocab index stays in [0, 51] for all valid and out-of-range inputs."""
    from hashi_puzzle_solver.models.encoders import _constraint_vocab_index, _CV_VOCAB_SIZE

    deg = torch.tensor([1, 2, 3, 4])
    nc = torch.tensor([0, 3, -1, 8])
    idx = _constraint_vocab_index(deg, nc)
    assert (idx >= 0).all() and (idx < _CV_VOCAB_SIZE).all()

    # Out-of-range inputs must clamp without error
    deg_oob = torch.tensor([0, 5])
    nc_oob = torch.tensor([-10, 20])
    idx_oob = _constraint_vocab_index(deg_oob, nc_oob)
    assert (idx_oob >= 0).all() and (idx_oob < _CV_VOCAB_SIZE).all()


def test_constraint_vocab_validation_error():
    """Enabling constraint vocab alongside individual embeddings raises ValueError."""
    import pytest

    config = ModelConfig(
        use_constraint_vocab=True,
        use_capacity=True,  # conflict
        use_structural_degree=False,
        use_structural_degree_nsew=False,
        use_unused_capacity=False,
        use_conflict_status=False,
        use_closeness_centrality=False,
        use_articulation_points=False,
        use_spectral_features=False,
    )
    fm = NodeFeatureManager(config)
    with pytest.raises(ValueError, match="use_constraint_vocab"):
        NodeEncoder(config, fm)


def test_constraint_vocab_features_registered():
    """Feature manager registers degree and unused_capacity columns for vocab lookup."""
    config = ModelConfig(
        use_constraint_vocab=True,
        use_capacity=False,
        use_structural_degree=False,
        use_structural_degree_nsew=False,
        use_unused_capacity=False,
        use_conflict_status=False,
        use_closeness_centrality=False,
        use_articulation_points=False,
        use_spectral_features=False,
    )
    fm = NodeFeatureManager(config)
    assert fm.has_feature("structural_degree")
    assert fm.has_feature("unused_capacity")
    assert fm.num_node_feats == 2
