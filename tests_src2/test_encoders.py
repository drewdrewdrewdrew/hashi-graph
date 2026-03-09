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
