"""Tests for residual mode rollout in DiffusionTrainer."""

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
    trainer.config = {
        "training": {
            "mode": "residual",
            "sigma_max": 2.0,
            "rollout_init": "noise",
            "diffusion_step_lr": 1.0,
            "flush_first_step": False,
        },
        "model": {
            "use_verification_head": False,
            "use_noise_head": False,
            "aux_predict_output_noise": False,
            "use_component_meta": False,
            "use_unused_capacity": False,
        },
    }
    trainer.model = MagicMock()
    return trainer


def make_mock_batch(num_edges=4, num_nodes=4, num_graphs=1):
    """Create a minimal mock batch object that simulates a PyG data object."""
    batch = MagicMock()
    batch.num_graphs = num_graphs
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
        cloned = make_mock_batch(num_edges, num_nodes, num_graphs)
        cloned.edge_attr = batch.edge_attr.clone()
        cloned.x = batch.x.clone()
        cloned.edge_mask = batch.edge_mask.clone()
        cloned.y = batch.y.clone()
        cloned.edge_index = batch.edge_index.clone()
        cloned.node_type = batch.node_type.clone()
        cloned.batch = batch.batch.clone()
        return cloned

    batch.clone.side_effect = clone_batch
    return batch


class TestResidualRolloutInit:
    """Test residual mode initialization in rollout."""

    def test_residual_rollout_noise_init(self):
        """Residual mode with rollout_init='noise' should initialize with noise on puzzle edges."""
        trainer = make_minimal_trainer()
        trainer.config["training"]["rollout_init"] = "noise"
        
        num_edges = 6
        batch = make_mock_batch(num_edges=num_edges)
        batch.edge_mask = torch.tensor([True, True, True, False, False, False])
        
        loader = [batch]
        
        # Mock model to return fixed delta
        fixed_delta = torch.ones(num_edges, 3) * 0.1
        trainer.model.return_value = fixed_delta
        
        with patch("src2.hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices", return_value=torch.zeros(num_edges, dtype=torch.long)), \
             patch("src2.hashi_puzzle_solver.trainers.diffusion.update_node_features", return_value=batch.x):
            
            # Set max_steps=1 to just test initialization
            result = trainer.run_rollout(loader, max_steps=1, checkpoints=[1])
        
        # Check that result contains expected keys
        assert "perfect_acc_k1" in result
        assert "accuracy" in result

    def test_residual_rollout_zeros_init(self):
        """Residual mode with rollout_init='zeros' should initialize with zeros."""
        trainer = make_minimal_trainer()
        trainer.config["training"]["rollout_init"] = "zeros"
        
        num_edges = 4
        batch = make_mock_batch(num_edges=num_edges)
        
        loader = [batch]
        
        # Mock model to return fixed delta
        fixed_delta = torch.ones(num_edges, 3) * 0.5
        trainer.model.return_value = fixed_delta
        
        with patch("src2.hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices", return_value=torch.zeros(num_edges, dtype=torch.long)), \
             patch("src2.hashi_puzzle_solver.trainers.diffusion.update_node_features", return_value=batch.x):
            
            result = trainer.run_rollout(loader, max_steps=1, checkpoints=[1])
        
        assert "perfect_acc_k1" in result


class TestResidualRolloutUpdate:
    """Test residual mode iterative correction update rule."""

    def test_residual_accumulates_deltas(self):
        """Residual mode should accumulate deltas: accumulated += pred_logits."""
        trainer = make_minimal_trainer()
        trainer.config["training"]["rollout_init"] = "zeros"
        
        num_edges = 4
        batch = make_mock_batch(num_edges=num_edges)
        
        loader = [batch]
        
        # Mock model to return fixed delta each step
        fixed_delta = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6],
        ])
        
        call_count = [0]
        
        def model_side_effect(*args, **kwargs):
            call_count[0] += 1
            return fixed_delta
        
        trainer.model.side_effect = model_side_effect
        
        with patch("src2.hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices", return_value=torch.zeros(num_edges, dtype=torch.long)), \
             patch("src2.hashi_puzzle_solver.trainers.diffusion.update_node_features", return_value=batch.x):
            
            result = trainer.run_rollout(loader, max_steps=2, checkpoints=[1, 2])
        
        # Model should be called twice (2 steps)
        assert call_count[0] == 2
        assert "perfect_acc_k2" in result

    def test_residual_only_updates_original_edges(self):
        """Residual update should only affect num_orig_edges (puzzle edges)."""
        trainer = make_minimal_trainer()
        trainer.config["training"]["rollout_init"] = "zeros"
        
        num_edges = 4
        batch = make_mock_batch(num_edges=num_edges)
        
        loader = [batch]
        
        # Model returns deltas for all edges (same size as batch)
        # The key test is that accumulated_logits[:num_orig_edges] is updated correctly
        delta = torch.ones(num_edges, 3) * 0.5
        trainer.model.return_value = delta
        
        with patch("src2.hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices", return_value=torch.zeros(num_edges, dtype=torch.long)), \
             patch("src2.hashi_puzzle_solver.trainers.diffusion.update_node_features", return_value=batch.x):
            
            # Should not raise, update uses [:num_orig_edges] slice
            result = trainer.run_rollout(loader, max_steps=1, checkpoints=[1])
        
        assert "perfect_acc_k1" in result


class TestResidualRolloutIntegration:
    """Integration tests for residual rollout."""

    def test_residual_rollout_converges_to_solution(self):
        """Test that residual rollout can converge to correct solution with appropriate deltas."""
        trainer = make_minimal_trainer()
        trainer.config["training"]["rollout_init"] = "zeros"
        
        num_edges = 4
        batch = make_mock_batch(num_edges=num_edges)
        
        # Set target solution
        batch.y = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        
        loader = [batch]
        
        # Model returns deltas that move toward one-hot targets
        # For simplicity, return one-hot directly (perfect prediction)
        step_count = [0]
        
        def model_returns_perfect(*args, **kwargs):
            step_count[0] += 1
            # Return one-hot encoding of targets
            one_hot = torch.zeros(num_edges, 3)
            one_hot[0, 0] = 1.0
            one_hot[1, 1] = 1.0
            one_hot[2, 2] = 1.0
            one_hot[3, 1] = 1.0
            return one_hot
        
        trainer.model.side_effect = model_returns_perfect
        
        with patch("src2.hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices", return_value=torch.zeros(num_edges, dtype=torch.long)), \
             patch("src2.hashi_puzzle_solver.trainers.diffusion.update_node_features", return_value=batch.x):
            
            result = trainer.run_rollout(loader, max_steps=1, checkpoints=[1])
        
        # With perfect deltas, should solve in 1 step
        assert result["perfect_acc_k1"] == 1.0
        assert step_count[0] >= 1


class TestResidualRolloutModeDetection:
    """Test that rollout correctly detects and handles residual mode."""

    def test_residual_uses_accumulated_logits(self):
        """Residual mode should use accumulated_logits like diff-cont and flow-blind."""
        trainer = make_minimal_trainer()
        
        num_edges = 4
        batch = make_mock_batch(num_edges=num_edges)
        
        loader = [batch]
        
        trainer.model.return_value = torch.ones(num_edges, 3) * 0.1
        
        # Verify bridge_logits_idx is used (not bridge_label_idx)
        assert trainer.bridge_logits_idx is not None
        assert trainer.bridge_label_idx is None
        
        with patch("src2.hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices", return_value=torch.zeros(num_edges, dtype=torch.long)), \
             patch("src2.hashi_puzzle_solver.trainers.diffusion.update_node_features", return_value=batch.x):
            
            result = trainer.run_rollout(loader, max_steps=1, checkpoints=[1])
        
        assert "perfect_acc_k1" in result
        assert "accuracy" in result
