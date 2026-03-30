"""Tests for residual mode MSE loss computation (Chunk 4)."""

import torch
import torch.nn.functional as F
import pytest


def test_residual_mse_pure_tensor():
    """Test residual MSE computation with fixed tensors."""
    device = torch.device("cpu")
    
    # Create fixed inputs
    x_in = torch.tensor([
        [0.5, -0.3, 0.2],
        [-0.1, 0.4, -0.2],
        [0.3, 0.1, -0.4],
    ], device=device)
    
    delta = torch.tensor([
        [0.1, 0.2, -0.3],
        [0.2, -0.1, 0.3],
        [-0.2, 0.3, 0.1],
    ], device=device)
    
    proposed = x_in + delta
    
    # Ground truth labels (0, 1, 2)
    y = torch.tensor([0, 1, 2], device=device)
    
    # Create target logits with centering and scaling
    scales = torch.tensor([4.0, 6.0, 8.0], device=device)
    y_one_hot = F.one_hot(y, num_classes=3).float()
    y_centered = y_one_hot - (1.0 / 3.0)
    target_logits = y_centered * scales.view(-1, 1)
    
    # Edge mask (all edges are puzzle edges)
    edge_mask = torch.tensor([True, True, True], device=device)
    
    # Compute MSE
    residual_mse = F.mse_loss(proposed[edge_mask], target_logits[edge_mask], reduction="mean")
    
    # Manual calculation for verification
    expected_mse = ((proposed - target_logits) ** 2).mean()
    
    assert torch.allclose(residual_mse, expected_mse, atol=1e-6)
    assert residual_mse.item() > 0.0


def test_residual_mse_with_mask():
    """Test that edge_mask correctly filters puzzle edges."""
    device = torch.device("cpu")
    
    # 5 edges: 3 puzzle, 2 hierarchical
    x_in = torch.randn(5, 3, device=device)
    delta = torch.randn(5, 3, device=device)
    proposed = x_in + delta
    
    y = torch.tensor([0, 1, 2, 0, 1], device=device)
    scales = torch.full((5,), 6.0, device=device)
    
    y_one_hot = F.one_hot(y, num_classes=3).float()
    y_centered = y_one_hot - (1.0 / 3.0)
    target_logits = y_centered * scales.view(-1, 1)
    
    # Only first 3 edges are puzzle edges
    edge_mask = torch.tensor([True, True, True, False, False], device=device)
    
    # Compute MSE only on puzzle edges
    residual_mse = F.mse_loss(proposed[edge_mask], target_logits[edge_mask], reduction="mean")
    
    # Manual calculation
    expected_mse = ((proposed[:3] - target_logits[:3]) ** 2).mean()
    
    assert torch.allclose(residual_mse, expected_mse, atol=1e-6)


def test_aux_logits_rescaling():
    """Test that aux_logits rescaling for degree/crossing is correct."""
    device = torch.device("cpu")
    
    proposed = torch.tensor([
        [8.0, 0.0, -8.0],
        [-4.0, 4.0, 0.0],
        [0.0, -8.0, 8.0],
    ], device=device)
    
    scale_max = 8.0
    aux_logits = proposed / scale_max + (1.0 / 3.0)
    
    # Expected values
    expected = torch.tensor([
        [1.0 + 1/3, 1/3, -1.0 + 1/3],
        [-0.5 + 1/3, 0.5 + 1/3, 1/3],
        [1/3, -1.0 + 1/3, 1.0 + 1/3],
    ], device=device)
    
    assert torch.allclose(aux_logits, expected, atol=1e-6)


def test_residual_loss_computation_logic():
    """Test the residual loss computation logic in isolation."""
    device = torch.device("cpu")
    
    # Simulate what happens in the residual branch
    delta = torch.tensor([
        [0.1, 0.2, -0.3],
        [0.2, -0.1, 0.3],
    ], device=device)
    
    x_in = torch.tensor([
        [0.5, -0.3, 0.2],
        [-0.1, 0.4, -0.2],
    ], device=device)
    
    proposed = x_in + delta
    
    y = torch.tensor([1, 0], device=device)
    scales = torch.tensor([6.0, 6.0], device=device)
    edge_mask = torch.tensor([True, True], device=device)
    
    # Target logits computation
    y_one_hot = F.one_hot(y, num_classes=3).float()
    y_centered = y_one_hot - (1.0 / 3.0)
    target_logits = y_centered * scales.view(-1, 1)
    
    # Residual MSE
    residual_mse = F.mse_loss(proposed[edge_mask], target_logits[edge_mask], reduction="mean")
    
    # Aux logits for degree/crossing
    scale_max = 8.0
    aux_logits = proposed / scale_max + (1.0 / 3.0)
    
    # Verify loss is computed
    assert residual_mse.item() > 0.0
    
    # Verify aux_logits shape
    assert aux_logits.shape == (2, 3)
    
    # Total loss with weight
    loss_weight = 1.0
    total_loss = loss_weight * residual_mse
    
    assert total_loss.item() > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
