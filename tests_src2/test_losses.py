"""Tests for modular losses."""

import torch
from hashi_puzzle_solver.models.config import HashiModelConfig
from hashi_puzzle_solver.losses.degree import DegreeLoss
from hashi_puzzle_solver.losses.crossing import CrossingLoss
from hashi_puzzle_solver.losses.calculator import HashiLossCalculator


def test_degree_loss():
    """Test that DegreeLoss correctly calculates violations."""
    loss_module = DegreeLoss()
    
    # Mock data: 2 nodes, bidirectional edge between them
    # Edge 0: 0->1, Edge 1: 1->0
    edge_index = torch.tensor([[0, 1], [1, 0]])
    node_capacities = torch.tensor([1, 1])
    edge_mask = torch.tensor([True, True])
    
    # Model predicts 1 bridge perfectly for both directions
    # Logits: [num_edges, 3] -> [2, 3]
    logits = torch.tensor([
        [0.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ]) 
    
    loss = loss_module(logits, edge_index, node_capacities, edge_mask)
    # Predicted degree will be ~1.0. Target is 1.0. Loss should be near 0.
    assert loss.item() < 1e-3

    # Model predicts 0 bridges
    logits = torch.tensor([
        [10.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    loss = loss_module(logits, edge_index, node_capacities, edge_mask)
    # Predicted degree ~0. Target is 1. MSE(0, 1) = 1.0
    # But wait, scatter sums per node. Node 0 has predicted degree 0. Node 1 has predicted degree 0.
    # MSE between [0, 0] and [1, 1] is ( (0-1)^2 + (0-1)^2 ) / 2 = 1.0
    assert abs(loss.item() - 1.0) < 1e-3


def test_crossing_loss():
    """Test that CrossingLoss penalizes crossing edges."""
    loss_module = CrossingLoss()
    
    # 2 edges crossing: edge 0 and edge 1
    edge_conflict_index = torch.tensor([[0], [1]])
    
    # Case 1: Only edge 0 has a bridge (Safe)
    logits = torch.tensor([
        [0.0, 10.0, 0.0], # Edge 0: Prob(bridge) ~ 1.0
        [10.0, 0.0, 0.0], # Edge 1: Prob(bridge) ~ 0.0
    ])
    loss = loss_module(logits, edge_conflict_index)
    # Loss ~ 1.0 * 0.0 = 0.0
    assert loss.item() < 1e-3
    
    # Case 2: Both edges have a bridge (Violation)
    logits = torch.tensor([
        [0.0, 10.0, 0.0], # Edge 0: Prob(bridge) ~ 1.0
        [0.0, 10.0, 0.0], # Edge 1: Prob(bridge) ~ 1.0
    ])
    loss = loss_module(logits, edge_conflict_index)
    # Loss ~ 1.0 * 1.0 = 1.0
    assert abs(loss.item() - 1.0) < 1e-3


def test_loss_calculator():
    """Test the integrated loss calculator."""
    config = HashiModelConfig()
    config.training.loss_weights.ce = 1.0
    config.training.loss_weights.degree = 0.5
    
    calculator = HashiLossCalculator(config)
    
    num_nodes = 2
    num_edges = 1
    logits = torch.randn(num_edges, 3)
    targets = torch.tensor([1])
    edge_index = torch.tensor([[0], [1]])
    node_capacities = torch.tensor([1, 1])
    edge_mask = torch.tensor([True])
    
    losses = calculator(logits, targets, edge_index, node_capacities, None, edge_mask)
    
    assert "total" in losses
    assert "ce" in losses
    assert "degree" in losses
    assert "crossing" in losses
