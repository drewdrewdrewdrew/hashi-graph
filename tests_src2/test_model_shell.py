"""Tests for HashiGraphModel shell."""

import torch
from hashi_puzzle_solver.models.config import HashiModelConfig
from hashi_puzzle_solver.models.features import NodeFeatureManager, EdgeFeatureManager
from hashi_puzzle_solver.models.encoders import NodeEncoder, EdgeEncoder
from hashi_puzzle_solver.models.backbone import GraphBackbone
from hashi_puzzle_solver.models.heads import EdgeHead, ProphetHead
from hashi_puzzle_solver.models.core import HashiGraphModel


def test_full_model_forward():
    """Test full forward pass of the integrated model shell."""
    config = HashiModelConfig()
    config.model.node_embedding_dim = 16
    config.model.hidden_channels = 32
    config.model.use_noise_head = True
    # Explicitly set some features for the test
    config.model.use_capacity = True
    config.model.use_unused_capacity = False
    config.model.use_structural_degree = False
    config.model.use_conflict_status = False
    config.model.use_closeness_centrality = False
    config.model.use_articulation_points = False
    config.model.use_spectral_features = False

    nm = NodeFeatureManager(config.model)
    em = EdgeFeatureManager(config.model)
    
    node_enc = NodeEncoder(config.model, nm)
    edge_enc = EdgeEncoder(config.model, em)
    
    # Calculate the dimension after EdgeEncoder
    encoded_edge_dim = edge_enc.output_dim
    
    backbone = GraphBackbone(
        node_input_dim=32,
        hidden_channels=32,
        num_layers=2,
        edge_dim=encoded_edge_dim,
    )
    
    edge_head = EdgeHead(config.model, 32, encoded_edge_dim)
    prophet_head = ProphetHead(config.model, 32)
    
    model = HashiGraphModel(
        config,
        node_enc,
        edge_enc,
        backbone,
        edge_head,
        prophet_head
    )
    
    # Mock data
    num_nodes = 10
    num_edges = 20
    x = torch.zeros((num_nodes, 1)) # Only capacity
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, em.num_edge_feats)
    node_type = torch.tensor([9] + [1] * (num_nodes - 1))
    
    # Forward pass
    logits = model(x, edge_index, edge_attr=edge_attr, node_type=node_type)
    assert logits.shape == (num_edges, 3)
    
    # Forward pass with noise head
    logits, noise_logits = model(
        x, 
        edge_index, 
        edge_attr=edge_attr, 
        node_type=node_type,
        return_noise=True
    )
    assert logits.shape == (num_edges, 3)
    assert noise_logits.shape == (1, 2)
