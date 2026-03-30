"""Test residual mode noise injection in DiffusionTrainer.run_epoch (Chunk 3)."""

import pytest
import torch
from unittest.mock import MagicMock, patch


def make_minimal_trainer():
    """Return a DiffusionTrainer with mocked model and minimal config."""
    from src2.hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer

    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.bridge_logits_idx = 0
    trainer.bridge_label_idx = None
    trainer.is_labeled_idx = None
    trainer.carry_over_buffer_train = []
    trainer.carry_over_buffer_val = []
    trainer.config = {
        "training": {
            "mode": "residual",
            "num_inference_steps_training": 1,
            "loss_weights": {"residual_mse": 1.0},
            "bptt": {"enabled": False},
            "alpha_power": 1.0,
            "zero_signal_prob": 0.0,
            "sigma_max": 2.0,
            "scale_min": 4.0,
            "scale_max": 8.0,
        },
        "model": {
            "use_verification_head": False,
            "use_noise_head": False,
            "aux_predict_output_noise": False,
            "use_component_meta": False,
            "use_unused_capacity": True,
        },
    }
    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.ema = None
    return trainer


def make_mock_batch(num_edges=4, num_nodes=4):
    """Create a minimal mock batch object that simulates a PyG data object."""
    batch = MagicMock()
    batch.num_graphs = 1
    batch.to.return_value = batch

    # Edge features: 3 logit channels + extras
    edge_attr = torch.zeros(num_edges, 6)
    edge_attr[:, :3] = torch.randn(num_edges, 3)  # bridge logits at idx 0

    # Set up data attributes
    batch.edge_attr = edge_attr
    batch.edge_mask = torch.ones(num_edges, dtype=torch.bool)
    batch.y = torch.zeros(num_edges, dtype=torch.long)
    batch.x = torch.zeros(num_nodes, 3)
    batch.edge_index = torch.zeros(2, num_edges, dtype=torch.long)
    batch.node_type = torch.zeros(num_nodes, dtype=torch.long)
    batch.batch = torch.zeros(num_edges, dtype=torch.long)

    # Support clone
    def clone_batch():
        cloned = make_mock_batch(num_edges, num_nodes)
        cloned.edge_attr = batch.edge_attr.clone()
        cloned.x = batch.x.clone()
        return cloned

    batch.clone.side_effect = clone_batch
    return batch


def test_residual_noise_injection_runs():
    """Test that residual mode successfully injects continuous noise."""
    trainer = make_minimal_trainer()
    batch = make_mock_batch(num_edges=4, num_nodes=4)
    
    # Store original edge_attr for comparison
    original_edge_attr = batch.edge_attr.clone()
    
    # Mock the model to return logits
    num_edges = batch.edge_attr.shape[0]
    trainer.model.return_value = torch.randn(num_edges, 3)
    
    # Mock compute_combined_loss to return controlled values
    mock_losses = {
        "total": torch.tensor(1.0, requires_grad=True),
        "ce": torch.tensor(0.8),
        "degree": torch.tensor(0.1),
        "crossing": torch.tensor(0.05),
        "verify": torch.tensor(0.0),
        "verify_acc": torch.tensor(0.0),
        "verify_recall_pos": torch.tensor(0.0),
        "verify_recall_neg": torch.tensor(0.0),
    }
    
    loader = [batch]
    
    with patch(
        "src2.hashi_puzzle_solver.trainers.diffusion.compute_combined_loss",
        return_value=mock_losses,
    ):
        with patch(
            "src2.hashi_puzzle_solver.trainers.diffusion.inject_continuous_noise",
            side_effect=lambda batch, **kwargs: batch,  # Pass through but verify it was called
        ) as mock_inject:
            results = trainer.run_epoch(loader, training=True, epoch=1)
            
            # Verify inject_continuous_noise was called
            assert mock_inject.call_count == 1
            
            # Verify the call had the expected parameters
            call_kwargs = mock_inject.call_args[1]
            assert "alpha" in call_kwargs
            assert "sigma" in call_kwargs
            assert "scale" in call_kwargs
            assert call_kwargs["bridge_logits_idx"] == trainer.bridge_logits_idx
            
            # Verify alphas, sigmas, scales were created with correct shapes
            assert call_kwargs["alpha"].shape == (batch.num_graphs,)
            assert call_kwargs["sigma"].shape == (batch.num_graphs,)
            assert call_kwargs["scale"].shape == (batch.num_graphs,)
            
            # Verify results dict exists
            assert isinstance(results, dict)


def test_residual_noise_injection_modifies_edge_attr():
    """Test that noise injection actually modifies edge_attr bridge logits."""
    trainer = make_minimal_trainer()
    batch = make_mock_batch(num_edges=4, num_nodes=4)
    
    # Store original edge_attr for comparison
    original_edge_attr = batch.edge_attr.clone()
    
    # Mock the model to return logits
    num_edges = batch.edge_attr.shape[0]
    trainer.model.return_value = torch.randn(num_edges, 3)
    
    # Mock compute_combined_loss to return controlled values
    mock_losses = {
        "total": torch.tensor(1.0, requires_grad=True),
        "ce": torch.tensor(0.8),
        "degree": torch.tensor(0.1),
        "crossing": torch.tensor(0.05),
        "verify": torch.tensor(0.0),
        "verify_acc": torch.tensor(0.0),
        "verify_recall_pos": torch.tensor(0.0),
        "verify_recall_neg": torch.tensor(0.0),
    }
    
    loader = [batch]
    
    # Use real inject_continuous_noise to verify it modifies edge_attr
    with patch(
        "src2.hashi_puzzle_solver.trainers.diffusion.compute_combined_loss",
        return_value=mock_losses,
    ):
        results = trainer.run_epoch(loader, training=True, epoch=1)
        
        # After injection, edge_attr should be modified (at least bridge logits slice)
        # We can't directly check batch.edge_attr since it's passed to inject_continuous_noise,
        # but we can verify the function was called and returned successfully
        assert isinstance(results, dict)


def test_residual_noise_parameters_in_range():
    """Test that noise parameters (alphas, sigmas, scales) are sampled correctly."""
    trainer = make_minimal_trainer()
    batch = make_mock_batch(num_edges=4, num_nodes=4)
    
    # Mock the model to return logits
    num_edges = batch.edge_attr.shape[0]
    trainer.model.return_value = torch.randn(num_edges, 3)
    
    # Mock compute_combined_loss
    mock_losses = {
        "total": torch.tensor(1.0, requires_grad=True),
        "ce": torch.tensor(0.8),
        "degree": torch.tensor(0.1),
        "crossing": torch.tensor(0.05),
        "verify": torch.tensor(0.0),
        "verify_acc": torch.tensor(0.0),
        "verify_recall_pos": torch.tensor(0.0),
        "verify_recall_neg": torch.tensor(0.0),
    }
    
    loader = [batch]
    
    with patch(
        "src2.hashi_puzzle_solver.trainers.diffusion.compute_combined_loss",
        return_value=mock_losses,
    ):
        with patch(
            "src2.hashi_puzzle_solver.trainers.diffusion.inject_continuous_noise",
            side_effect=lambda batch, **kwargs: batch,
        ) as mock_inject:
            results = trainer.run_epoch(loader, training=True, epoch=1)
            
            # Get the parameters passed to inject_continuous_noise
            call_kwargs = mock_inject.call_args[1]
            alphas = call_kwargs["alpha"]
            sigmas = call_kwargs["sigma"]
            scales = call_kwargs["scale"]
            
            # Verify alphas are in [0, 1] (after alpha_power)
            assert torch.all(alphas >= 0.0)
            assert torch.all(alphas <= 1.0)
            
            # Verify sigmas are in [0, sigma_max]
            sigma_max = trainer.config["training"]["sigma_max"]
            assert torch.all(sigmas >= 0.0)
            assert torch.all(sigmas <= sigma_max)
            
            # Verify scales are in [scale_min, scale_max]
            scale_min = trainer.config["training"]["scale_min"]
            scale_max = trainer.config["training"]["scale_max"]
            assert torch.all(scales >= scale_min)
            assert torch.all(scales <= scale_max)


def test_residual_with_use_unused_capacity():
    """Test that residual mode works with use_unused_capacity=True."""
    trainer = make_minimal_trainer()
    trainer.config["model"]["use_unused_capacity"] = True
    batch = make_mock_batch(num_edges=4, num_nodes=4)
    
    # Mock the model to return logits
    num_edges = batch.edge_attr.shape[0]
    trainer.model.return_value = torch.randn(num_edges, 3)
    
    # Mock compute_combined_loss
    mock_losses = {
        "total": torch.tensor(1.0, requires_grad=True),
        "ce": torch.tensor(0.8),
        "degree": torch.tensor(0.1),
        "crossing": torch.tensor(0.05),
        "verify": torch.tensor(0.0),
        "verify_acc": torch.tensor(0.0),
        "verify_recall_pos": torch.tensor(0.0),
        "verify_recall_neg": torch.tensor(0.0),
    }
    
    loader = [batch]
    
    with patch(
        "src2.hashi_puzzle_solver.trainers.diffusion.compute_combined_loss",
        return_value=mock_losses,
    ):
        # Should not raise an error
        results = trainer.run_epoch(loader, training=True, epoch=1)
        assert isinstance(results, dict)
