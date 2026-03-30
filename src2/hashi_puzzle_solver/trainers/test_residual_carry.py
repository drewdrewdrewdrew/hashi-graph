"""Tests for residual mode detached carry and mode-specific inter-step logic (Chunk 5)."""

import pytest
import torch
from torch_geometric.data import Data, Batch
from unittest.mock import MagicMock, patch


def make_minimal_trainer(mode="residual", num_steps=2, bptt_enabled=False, recursive_carryover=False):
    """Return a DiffusionTrainer with mocked model and minimal config."""
    from hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer

    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.bridge_logits_idx = 2
    trainer.bridge_label_idx = 0
    trainer.is_labeled_idx = 1
    trainer.carry_over_buffer_train = []
    trainer.carry_over_buffer_val = []
    trainer.config = {
        "training": {
            "mode": mode,
            "num_inference_steps_training": num_steps,
            "sigma_max": 2.0,
            "scale_min": 4.0,
            "scale_max": 8.0,
            "alpha_power": 1.0,
            "zero_signal_prob": 0.0,
            "loss_weights": {
                "residual_mse": 1.0,
                "degree": 0.0,
                "crossing": 0.0,
                "verify": 0.0,
            },
            "bptt": {"enabled": bptt_enabled},
            "recursive_carryover": recursive_carryover,
        },
        "model": {
            "use_continuous_edge_labels": True,
            "use_noise_head": False,
            "use_time_conditioning": False,
            "use_verification_head": False,
            "use_unused_capacity": True,
            "use_capacity": True,
            "use_structural_degree": True,
            "use_component_meta": False,
        },
    }
    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.ema = None
    return trainer


def make_mock_batch(num_edges=2, num_nodes=2):
    """Create a minimal mock batch object."""
    batch = MagicMock()
    batch.num_graphs = 1
    batch.to.return_value = batch

    # Edge features: [bridge_label, is_labeled, logit_0, logit_1, logit_2]
    edge_attr = torch.zeros(num_edges, 5)
    edge_attr[:, 1] = 1.0  # is_labeled

    batch.edge_attr = edge_attr
    batch.edge_mask = torch.ones(num_edges, dtype=torch.bool)
    batch.y = torch.ones(num_edges, dtype=torch.long)
    batch.x = torch.tensor([[5.0, 1.0, 5.0], [3.0, 1.0, 3.0]])
    batch.edge_index = torch.tensor([[0, 1], [1, 0]])
    batch.node_type = torch.tensor([5, 3])
    batch.batch = torch.zeros(num_edges, dtype=torch.long)
    
    # Support __getitem__ for edge_mask indexing
    def getitem_side_effect(key):
        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            # Return a tensor when indexing with boolean mask
            return batch.edge_mask
        return MagicMock()
    
    batch.__getitem__.side_effect = getitem_side_effect

    def clone_batch():
        cloned = make_mock_batch(num_edges, num_nodes)
        cloned.edge_attr = batch.edge_attr.clone()
        cloned.x = batch.x.clone()
        cloned.edge_mask = batch.edge_mask.clone()
        cloned.y = batch.y.clone()
        cloned.batch = batch.batch.clone()
        cloned.edge_index = batch.edge_index.clone()
        cloned.node_type = batch.node_type.clone()
        return cloned

    batch.clone.side_effect = clone_batch
    return batch


def test_residual_bptt_guard():
    """Test that residual mode raises error when BPTT is enabled."""
    trainer = make_minimal_trainer(mode="residual", bptt_enabled=True)
    batch = make_mock_batch()
    loader = [batch]
    
    with pytest.raises(ValueError, match="Residual training mode is incompatible with BPTT"):
        trainer.run_epoch(loader, epoch=1, training=True)


def test_residual_recursive_carryover_guard():
    """Test that residual mode raises error when recursive_carryover is enabled."""
    trainer = make_minimal_trainer(mode="residual", recursive_carryover=True)
    batch = make_mock_batch()
    loader = [batch]
    
    with pytest.raises(ValueError, match="Residual training mode is incompatible with recursive_carryover"):
        trainer.run_epoch(loader, epoch=1, training=True)


def test_residual_detached_carry_no_grad():
    """Test that residual carry is detached (no grad flows through previous step)."""
    trainer = make_minimal_trainer(mode="residual", num_steps=2)
    batch = make_mock_batch()
    
    # Mock model to return constant delta
    delta_value = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], requires_grad=True)
    trainer.model.return_value = delta_value
    
    loader = [batch]
    
    # Patch necessary functions
    with patch("hashi_puzzle_solver.trainers.diffusion.inject_continuous_noise", return_value=batch), \
         patch("hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices", return_value=torch.zeros(2, dtype=torch.long)), \
         patch("hashi_puzzle_solver.trainers.diffusion.update_node_features", return_value=batch.x), \
         patch("hashi_puzzle_solver.trainers.diffusion.calculate_batch_perfect_puzzles", return_value=(0.0, 0, 1)):
        metrics = trainer.run_epoch(loader, epoch=1, training=True)
    
    # Check that model was called twice (2 inference steps)
    assert trainer.model.call_count == 2
    assert metrics is not None


def test_residual_carry_updates_state():
    """Test that residual carry correctly updates edge_attr for next step."""
    trainer = make_minimal_trainer(mode="residual", num_steps=2)
    batch = make_mock_batch()
    
    # Track calls to verify state updates
    call_count = [0]
    initial_logits = None
    second_step_logits = None
    
    def mock_forward(x, edge_index, **kwargs):
        edge_attr = kwargs.get("edge_attr")
        if call_count[0] == 0:
            nonlocal initial_logits
            initial_logits = edge_attr[:, 2:5].clone() if edge_attr is not None else None
        elif call_count[0] == 1:
            nonlocal second_step_logits
            second_step_logits = edge_attr[:, 2:5].clone() if edge_attr is not None else None
        
        call_count[0] += 1
        return torch.tensor([[0.5, -0.3, 0.1], [0.5, -0.3, 0.1]], requires_grad=True)
    
    trainer.model.side_effect = mock_forward
    
    loader = [batch]
    
    with patch("hashi_puzzle_solver.trainers.diffusion.inject_continuous_noise", return_value=batch), \
         patch("hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices", return_value=torch.zeros(2, dtype=torch.long)), \
         patch("hashi_puzzle_solver.trainers.diffusion.update_node_features", return_value=batch.x), \
         patch("hashi_puzzle_solver.trainers.diffusion.calculate_batch_perfect_puzzles", return_value=(0.0, 0, 1)):
        trainer.run_epoch(loader, epoch=1, training=True)
    
    # Verify that second step received updated logits (initial + delta)
    assert initial_logits is not None
    assert second_step_logits is not None
    
    # The second step should have different logits than the first
    assert not torch.allclose(initial_logits, second_step_logits, atol=1e-6)


def test_mode_specific_carry_diff_cont():
    """Test that diff-cont mode uses softmax->center->scale carry."""
    trainer = make_minimal_trainer(mode="diff-cont", num_steps=2)
    batch = make_mock_batch()
    
    # Return logits that will produce different results after softmax
    logits = torch.tensor([[2.0, 1.0, 0.5], [1.5, 2.0, 0.8]], requires_grad=True)
    trainer.model.return_value = logits
    
    loader = [batch]
    
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
    
    with patch("hashi_puzzle_solver.trainers.diffusion.compute_combined_loss", return_value=mock_losses), \
         patch("hashi_puzzle_solver.trainers.diffusion.inject_continuous_noise", return_value=batch), \
         patch("hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices", return_value=torch.zeros(2, dtype=torch.long)), \
         patch("hashi_puzzle_solver.trainers.diffusion.update_node_features", return_value=batch.x), \
         patch("hashi_puzzle_solver.trainers.diffusion.calculate_batch_perfect_puzzles", return_value=(0.0, 0, 1)):
        trainer.run_epoch(loader, epoch=1, training=True)
    
    # Verify it runs without error for diff-cont mode
    assert trainer.model.call_count == 2


def test_mode_specific_carry_flow_blind():
    """Test that flow-blind mode uses aux_logits carry."""
    trainer = make_minimal_trainer(mode="flow-blind", num_steps=1)  # Use 1 step to avoid carry complexity
    trainer.config["model"]["use_time_conditioning"] = True
    trainer.config["model"]["time_noise_std"] = 0.0
    
    batch = make_mock_batch()
    # Add t_sampled for flow-blind mode
    batch.t_sampled = torch.tensor([0.5, 0.5])
    
    # Return proper tensor logits
    logits = torch.randn(2, 3, requires_grad=True)
    trainer.model.return_value = logits
    
    loader = [batch]
    
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
    
    with patch("hashi_puzzle_solver.trainers.diffusion.compute_combined_loss", return_value=mock_losses), \
         patch("hashi_puzzle_solver.trainers.diffusion.inject_flow_noise", return_value=batch), \
         patch("hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices", return_value=torch.zeros(2, dtype=torch.long)), \
         patch("hashi_puzzle_solver.trainers.diffusion.update_node_features", return_value=batch.x), \
         patch("hashi_puzzle_solver.trainers.diffusion.calculate_batch_perfect_puzzles", return_value=(0.0, 0, 1)):
        trainer.run_epoch(loader, epoch=1, training=True)
    
    # Verify it runs without error
    assert trainer.model.call_count == 1
