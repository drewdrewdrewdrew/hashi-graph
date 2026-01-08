"""
Unit tests for AR loss functions.
"""
import torch
from hashi_puzzle_solver.losses import asymmetric_mse_loss, ordinal_bce_loss, compute_ar_loss


class TestAsymmetricMSELoss:
    """Test asymmetric MSE loss functionality."""

    def test_basic_functionality(self):
        """Test basic asymmetric MSE computation."""
        y_pred = torch.tensor([1.5, 2.5, 0.5])
        y_true = torch.tensor([1.0, 2.0, 1.0])

        loss = asymmetric_mse_loss(y_pred, y_true, alpha=2.0)

        # Calculate expected losses manually:
        # diff: [0.5, 0.5, -0.5]
        # overshoot: 0.5^2 * 2 = 0.5, undershoot: 0.5^2 = 0.25
        # expected: (0.5 + 0.5 + 0.25) / 3 = 0.4167
        expected = (0.5 + 0.5 + 0.25) / 3.0

        assert abs(loss.item() - expected) < 1e-6

    def test_no_penalty_undershoot(self):
        """Test that undershooting gets normal MSE."""
        y_pred = torch.tensor([0.5, 0.8])
        y_true = torch.tensor([1.0, 1.0])

        loss = asymmetric_mse_loss(y_pred, y_true, alpha=10.0)

        # Both are undershoots, so normal MSE
        expected = ((0.5-1.0)**2 + (0.8-1.0)**2) / 2.0
        assert abs(loss.item() - expected) < 1e-6

    def test_heavy_penalty_overshoot(self):
        """Test that overshooting gets heavy penalty."""
        y_pred = torch.tensor([1.5, 2.2])
        y_true = torch.tensor([1.0, 2.0])

        loss = asymmetric_mse_loss(y_pred, y_true, alpha=3.0)

        # diff: [0.5, 0.2] -> overshoot penalty: 0.5^2 * 3 + 0.2^2 * 3
        expected = (0.5**2 * 3 + 0.2**2 * 3) / 2.0
        assert abs(loss.item() - expected) < 1e-6


class TestOrdinalBCELoss:
    """Test ordinal BCE loss functionality."""

    def test_target_conversion(self):
        """Test that targets are converted correctly."""
        # Target 0 -> [0, 0]
        # Target 1 -> [1, 0]
        # Target 2 -> [1, 1]
        logits = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        targets = torch.tensor([0, 1, 2])

        loss = ordinal_bce_loss(logits, targets)

        # Should compute BCE with targets [[0,0], [1,0], [1,1]]
        assert loss.item() > 0  # Just check it's a valid loss


class TestComputeARLoss:
    """Test the main AR loss function."""

    def test_regression_head(self):
        """Test AR loss with regression head."""
        output = torch.tensor([1.5, 0.8, 2.1])
        targets = torch.tensor([1.0, 1.0, 2.0])  # Remaining capacity
        current_bridges = torch.tensor([0, 1, 0])
        edge_mask = torch.tensor([True, True, True])

        loss = compute_ar_loss(
            output, targets, current_bridges, edge_mask,
            head_type='regression', overshoot_penalty=2.0
        )

        # Should use asymmetric MSE
        assert loss.item() > 0

    def test_conditional_head(self):
        """Test AR loss with conditional head."""
        output = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # [batch, 2]
        targets = torch.tensor([0, 1, 2])  # Remaining capacity
        current_bridges = torch.tensor([0, 1, 0])
        edge_mask = torch.tensor([True, True, True])

        loss = compute_ar_loss(
            output, targets, current_bridges, edge_mask,
            head_type='conditional'
        )

        # Should use ordinal BCE
        assert loss.item() > 0

    def test_locked_edges_ignored(self):
        """Test that locked edges (current_bridges >= 2) are ignored."""
        output = torch.tensor([1.5, 0.8])
        targets = torch.tensor([1.0, 1.0])
        current_bridges = torch.tensor([0, 2])  # Second edge is locked
        edge_mask = torch.tensor([True, True])

        loss = compute_ar_loss(
            output, targets, current_bridges, edge_mask,
            head_type='regression'
        )

        # Should only compute loss on first edge (second is locked)
        # diff = 1.5 - 1.0 = 0.5 (overshoot)
        # loss = 5.0 * (0.5)^2 = 1.25 (alpha=5.0 penalty for overshooting)
        expected = 5.0 * (0.5**2)  # Only one edge contributes

        assert abs(loss.item() - expected) < 1e-6