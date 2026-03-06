"""BPTT window loop tests for DiffusionTrainer."""

import pytest
import torch
from unittest.mock import MagicMock, patch


def make_bptt_trainer(window=2, stride=1, num_steps=3):
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
            "mode": "diff-cont",
            "num_inference_steps_training": num_steps,
            "loss_weights": {"ce": 1.0, "degree": 0.0, "crossing": 0.0, "verify": 0.0, "noise": 0.0},
            "bptt": {"enabled": True, "window": window, "stride": stride, "loss_ema_decay": 0.9},
        },
        "model": {
            "use_verification_head": False,
            "use_noise_head": False,
            "aux_predict_output_noise": False,
            "use_component_meta": False,
            "use_unused_capacity": False,
        },
    }
    # Model returns a real parameter-backed tensor so autograd can flow
    param = torch.nn.Parameter(torch.zeros(4, 3))
    trainer.model = MagicMock(return_value=param)
    trainer.optimizer = MagicMock()
    return trainer, param


def make_bptt_batch(num_edges=4, num_nodes=4):
    """Create a minimal mock batch suitable for _run_bptt_window calls."""
    batch = MagicMock()
    batch.num_graphs = 1
    batch.to.return_value = batch

    edge_attr = torch.zeros(num_edges, 6)
    edge_attr[:, :3] = torch.randn(num_edges, 3)

    batch.edge_attr = edge_attr
    batch.edge_mask = torch.ones(num_edges, dtype=torch.bool)
    batch.y = torch.zeros(num_edges, dtype=torch.long)
    batch.x = torch.zeros(num_nodes, 3)
    batch.edge_index = torch.zeros(2, num_edges, dtype=torch.long)
    batch.node_type = torch.zeros(num_nodes, dtype=torch.long)
    batch.batch = torch.zeros(num_edges, dtype=torch.long)

    def clone_batch():
        cloned = make_bptt_batch(num_edges, num_nodes)
        cloned.edge_attr = batch.edge_attr.clone()
        cloned.x = batch.x.clone()
        return cloned

    batch.clone.side_effect = clone_batch
    return batch


# ---------------------------------------------------------------------------
# Task 1 tests: _run_bptt_window returns differentiable scalar
# ---------------------------------------------------------------------------

class TestRunBpttWindowReturnsScalar:
    """_run_bptt_window returns a 0-dim tensor."""

    def test_run_bptt_window_returns_scalar(self):
        trainer, param = make_bptt_trainer(window=2, stride=1, num_steps=3)
        batch = make_bptt_batch()
        scales = torch.ones(1)
        num_edges = 4
        step_boundary_states = [
            torch.randn(num_edges, 3) for _ in range(3)
        ]

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

        with patch(
            "src2.hashi_puzzle_solver.trainers.diffusion.compute_combined_loss",
            return_value=mock_losses,
        ), patch(
            "src2.hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices",
            return_value=torch.zeros(num_edges, dtype=torch.long),
        ), patch(
            "src2.hashi_puzzle_solver.trainers.diffusion.update_node_features",
            return_value=batch.x,
        ):
            result = trainer._run_bptt_window(
                start_data=batch,
                start_step=0,
                end_step=2,
                step_boundary_states=step_boundary_states,
                batch=batch,
                scales=scales,
                training_cfg=trainer.config["training"],
                loss_weights=trainer.config["training"]["loss_weights"],
            )

        assert result.dim() == 0, f"Expected 0-dim tensor, got {result.dim()}-dim"


class TestRunBpttWindowHasGrad:
    """_run_bptt_window returned tensor has requires_grad=True."""

    def test_run_bptt_window_has_grad(self):
        trainer, param = make_bptt_trainer(window=1, stride=1, num_steps=1)
        batch = make_bptt_batch()
        scales = torch.ones(1)
        num_edges = 4
        step_boundary_states = [torch.randn(num_edges, 3)]

        mock_losses = {
            "total": param.sum(),  # differentiable through param
            "ce": torch.tensor(0.8),
            "degree": torch.tensor(0.1),
            "crossing": torch.tensor(0.05),
            "verify": torch.tensor(0.0),
            "verify_acc": torch.tensor(0.0),
            "verify_recall_pos": torch.tensor(0.0),
            "verify_recall_neg": torch.tensor(0.0),
        }

        with patch(
            "src2.hashi_puzzle_solver.trainers.diffusion.compute_combined_loss",
            return_value=mock_losses,
        ), patch(
            "src2.hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices",
            return_value=torch.zeros(num_edges, dtype=torch.long),
        ), patch(
            "src2.hashi_puzzle_solver.trainers.diffusion.update_node_features",
            return_value=batch.x,
        ):
            result = trainer._run_bptt_window(
                start_data=batch,
                start_step=0,
                end_step=1,
                step_boundary_states=step_boundary_states,
                batch=batch,
                scales=scales,
                training_cfg=trainer.config["training"],
                loss_weights=trainer.config["training"]["loss_weights"],
            )

        assert result.requires_grad, "Returned tensor must have requires_grad=True"


class TestWindowLossBackwardDoesNotRaise:
    """.backward() on the _run_bptt_window result does not raise."""

    def test_window_loss_backward_does_not_raise(self):
        trainer, param = make_bptt_trainer(window=1, stride=1, num_steps=1)
        batch = make_bptt_batch()
        scales = torch.ones(1)
        num_edges = 4
        step_boundary_states = [torch.randn(num_edges, 3)]

        mock_losses = {
            "total": param.sum(),
            "ce": torch.tensor(0.8),
            "degree": torch.tensor(0.1),
            "crossing": torch.tensor(0.05),
            "verify": torch.tensor(0.0),
            "verify_acc": torch.tensor(0.0),
            "verify_recall_pos": torch.tensor(0.0),
            "verify_recall_neg": torch.tensor(0.0),
        }

        with patch(
            "src2.hashi_puzzle_solver.trainers.diffusion.compute_combined_loss",
            return_value=mock_losses,
        ), patch(
            "src2.hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices",
            return_value=torch.zeros(num_edges, dtype=torch.long),
        ), patch(
            "src2.hashi_puzzle_solver.trainers.diffusion.update_node_features",
            return_value=batch.x,
        ):
            result = trainer._run_bptt_window(
                start_data=batch,
                start_step=0,
                end_step=1,
                step_boundary_states=step_boundary_states,
                batch=batch,
                scales=scales,
                training_cfg=trainer.config["training"],
                loss_weights=trainer.config["training"]["loss_weights"],
            )

        # Should not raise
        result.backward()

        assert param.grad is not None, "param.grad must be non-None after .backward()"


# ---------------------------------------------------------------------------
# Task 2 tests: window loop wired into run_epoch
# ---------------------------------------------------------------------------

def make_mock_loader(batch):
    return [batch]


def _make_full_mock_losses(requires_grad=True):
    return {
        "total": torch.tensor(1.0, requires_grad=requires_grad),
        "ce": torch.tensor(0.8),
        "degree": torch.tensor(0.1),
        "crossing": torch.tensor(0.05),
        "verify": torch.tensor(0.0),
        "verify_acc": torch.tensor(0.0),
        "verify_recall_pos": torch.tensor(0.0),
        "verify_recall_neg": torch.tensor(0.0),
    }


def _run_bptt_epoch(trainer, batch, num_steps, window, stride):
    """Run run_epoch with bptt enabled, patching _run_bptt_window to return simple tensors."""
    trainer.config["training"]["num_inference_steps_training"] = num_steps
    trainer.config["training"]["bptt"] = {
        "enabled": True,
        "window": window,
        "stride": stride,
        "loss_ema_decay": 0.9,
    }
    num_edges = batch.edge_attr.shape[0]
    loader = make_mock_loader(batch)

    # Each call to _run_bptt_window returns a fresh differentiable tensor
    def fake_bptt_window(*args, **kwargs):
        return torch.tensor(1.0, requires_grad=True)

    with patch(
        "src2.hashi_puzzle_solver.trainers.diffusion.compute_combined_loss",
        return_value=_make_full_mock_losses(),
    ), patch(
        "src2.hashi_puzzle_solver.trainers.diffusion.inject_continuous_noise",
        return_value=batch,
    ), patch(
        "src2.hashi_puzzle_solver.trainers.diffusion.get_edge_batch_indices",
        return_value=torch.zeros(num_edges, dtype=torch.long),
    ), patch(
        "src2.hashi_puzzle_solver.trainers.diffusion.update_node_features",
        return_value=batch.x,
    ), patch(
        "src2.hashi_puzzle_solver.trainers.diffusion.calculate_batch_perfect_puzzles",
        return_value=(None, 0, 1),
    ), patch.object(
        trainer.__class__, "_run_bptt_window", side_effect=fake_bptt_window
    ) as mock_window:
        result = trainer.run_epoch(loader, training=True, epoch=1, total_epochs=1)

    return result, mock_window


class TestBpttWindowCountStride1:
    """num_steps=3, window=2, stride=1 -> 3 windows."""

    def test_bptt_window_count_stride1(self):
        from src2.hashi_puzzle_solver.trainers.test_diffusion_compat import make_mock_batch
        trainer, _ = make_bptt_trainer(window=2, stride=1, num_steps=3)
        batch = make_mock_batch()

        result, mock_window = _run_bptt_epoch(trainer, batch, num_steps=3, window=2, stride=1)

        # Windows: [0,2), [1,3), [2,3) -> 3 calls
        assert mock_window.call_count == 3, (
            f"Expected 3 _run_bptt_window calls, got {mock_window.call_count}"
        )
        # optimizer.step() called exactly once
        assert trainer.optimizer.step.call_count == 1, (
            f"Expected optimizer.step() called 1 time, got {trainer.optimizer.step.call_count}"
        )


class TestBpttWindowCountStride2:
    """num_steps=4, window=2, stride=2 -> 2 windows."""

    def test_bptt_window_count_stride2(self):
        from src2.hashi_puzzle_solver.trainers.test_diffusion_compat import make_mock_batch
        trainer, _ = make_bptt_trainer(window=2, stride=2, num_steps=4)
        batch = make_mock_batch()

        result, mock_window = _run_bptt_epoch(trainer, batch, num_steps=4, window=2, stride=2)

        # Windows: [0,2), [2,4) -> 2 calls
        assert mock_window.call_count == 2, (
            f"Expected 2 _run_bptt_window calls, got {mock_window.call_count}"
        )
        assert trainer.optimizer.step.call_count == 1, (
            f"Expected optimizer.step() called 1 time, got {trainer.optimizer.step.call_count}"
        )


class TestBpttEmaUpdates:
    """bptt_ema is a float in reported loss, not None."""

    def test_bptt_ema_updates(self):
        from src2.hashi_puzzle_solver.trainers.test_diffusion_compat import make_mock_batch
        trainer, _ = make_bptt_trainer(window=2, stride=1, num_steps=3)
        batch = make_mock_batch()

        result, _ = _run_bptt_epoch(trainer, batch, num_steps=3, window=2, stride=1)

        assert "loss" in result, "result must contain 'loss' key"
        loss_val = result["loss"]
        assert isinstance(loss_val, float), f"loss must be a float, got {type(loss_val)}"
        assert loss_val > 0.0, "EMA loss must be positive (window loss is 1.0)"


class TestBpttSingleWindowEquivalent:
    """num_steps=1, window=1, stride=1 -> one window, one backward, one optimizer.step()."""

    def test_bptt_single_window_equivalent(self):
        from src2.hashi_puzzle_solver.trainers.test_diffusion_compat import make_mock_batch
        trainer, _ = make_bptt_trainer(window=1, stride=1, num_steps=1)
        batch = make_mock_batch()

        result, mock_window = _run_bptt_epoch(trainer, batch, num_steps=1, window=1, stride=1)

        assert mock_window.call_count == 1, (
            f"Expected 1 _run_bptt_window call, got {mock_window.call_count}"
        )
        assert trainer.optimizer.step.call_count == 1, (
            f"Expected optimizer.step() called 1 time, got {trainer.optimizer.step.call_count}"
        )
