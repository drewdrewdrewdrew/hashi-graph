"""Tests for Backbone and Heads."""

import torch
from hashi_puzzle_solver.models.config import HashiModelConfig
from hashi_puzzle_solver.models.backbone import GraphBackbone
from hashi_puzzle_solver.models.heads import EdgeHead, ProphetHead


def test_backbone_forward():
    """Test that GraphBackbone works for different GNN types."""
    batch_size = 10
    num_nodes = 20
    num_edges = 50
    hidden_dim = 32
    
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4)
    h = torch.randn(num_nodes, hidden_dim)

    # Test Transformer
    backbone = GraphBackbone(
        node_input_dim=hidden_dim,
        hidden_channels=hidden_dim,
        num_layers=2,
        heads=4,
        edge_dim=4,
        gnn_type="transformer"
    )
    h_out = backbone(h, edge_index, edge_attr)
    assert h_out.shape == (num_nodes, hidden_dim)

    # Test GAT
    backbone_gat = GraphBackbone(
        node_input_dim=hidden_dim,
        hidden_channels=hidden_dim,
        num_layers=2,
        heads=4,
        edge_dim=4,
        gnn_type="gat"
    )
    h_out_gat = backbone_gat(h, edge_index, edge_attr)
    assert h_out_gat.shape == (num_nodes, hidden_dim)


def test_edge_head_forward():
    """Test that EdgeHead generates correct output shapes."""
    config = HashiModelConfig()
    config.model.edge_mlp_width_mult = 1.0
    config.model.edge_mlp_depth_mult = 1
    config.model.edge_concat_global_meta = True
    
    node_hidden_dim = 32
    edge_attr_dim = 4
    num_nodes = 10
    num_edges = 20
    
    h = torch.randn(num_nodes, node_hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_attr_dim)
    # 1 global meta node at index 0
    node_type = torch.tensor([9] + [1] * (num_nodes - 1))
    
    head = EdgeHead(config.model, node_hidden_dim, edge_attr_dim)
    logits = head(h, edge_index, edge_attr, node_type=node_type)
    
    assert logits.shape == (num_edges, 3)


def test_prophet_head_forward():
    """Test that ProphetHead generates correct output shapes."""
    config = HashiModelConfig()
    node_hidden_dim = 32
    num_nodes = 5
    num_edges = 10
    
    h = torch.randn(num_nodes, node_hidden_dim)
    edge_logits = torch.randn(num_edges, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    node_type = torch.tensor([9] + [1] * (num_nodes - 1))
    
    head = ProphetHead(config.model, node_hidden_dim)
    aux_logits = head(h, edge_logits, edge_index, node_type=node_type)
    
    assert aux_logits.shape == (1, 2)  # (sigma, alpha) for one graph
