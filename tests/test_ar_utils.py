"""
Unit tests for AR utility functions.
"""
import torch
from hashi_puzzle_solver.train_utils import (
    get_unused_capacity_index, update_node_features,
    apply_ar_action, select_ar_action
)
from hashi_puzzle_solver.models.heads import RegressionActionHead, ConditionalActionHead


class TestGetUnusedCapacityIndex:
    """Test index calculation for unused_capacity feature."""

    def test_basic_config(self):
        """Test with basic config (capacity + structural_degree + unused_capacity)."""
        config = {
            'use_capacity': True,
            'use_structural_degree': True,
            'use_unused_capacity': True
        }
        idx = get_unused_capacity_index(config)
        assert idx == 2  # After capacity (0) and structural_degree (1)

    def test_minimal_config(self):
        """Test with minimal config (just unused_capacity)."""
        config = {
            'use_capacity': False,
            'use_structural_degree': False,
            'use_unused_capacity': True
        }
        idx = get_unused_capacity_index(config)
        assert idx == 0  # First feature


class TestUpdateNodeFeatures:
    """Test node feature updates during AR training."""

    def test_update_unused_capacity(self):
        """Test that unused_capacity is correctly updated based on bridge state."""
        # Mock node features: [capacity, structural_degree, unused_capacity]
        x = torch.tensor([
            [4.0, 2.0, 4.0],  # Island with capacity 4
            [2.0, 1.0, 2.0],  # Island with capacity 2
            [0.0, 0.0, 0.0],  # Meta node
        ])

        # Mock edge connections and current bridges
        edge_index = torch.tensor([
            [0, 1],  # Edge between island 0 and 1
            [1, 2],  # Edge between island 1 and meta
        ])
        current_bridges = torch.tensor([1.0, 0.0])  # 1 bridge on first edge

        node_type = torch.tensor([1, 2, 9])  # Island, island, meta

        config = {
            'use_capacity': True,
            'use_structural_degree': True,
            'use_unused_capacity': True
        }

        updated_x = update_node_features(x, current_bridges, edge_index, node_type, config)

        # Island 0: original capacity 4, used 1 bridge -> remaining 3
        # Island 1: original capacity 2, used 1 bridge -> remaining 1
        # Meta: unchanged (0)
        expected = torch.tensor([
            [4.0, 2.0, 3.0],  # Island 0: 4 - 1 = 3
            [2.0, 1.0, 1.0],  # Island 1: 2 - 1 = 1
            [0.0, 0.0, 0.0],  # Meta: unchanged
        ])

        assert torch.allclose(updated_x, expected)

    def test_no_unused_capacity_feature(self):
        """Test that function returns unchanged features when unused_capacity disabled."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        current_bridges = torch.tensor([0.0])
        edge_index = torch.tensor([[0, 1]])
        node_type = torch.tensor([1, 2])

        config = {'use_unused_capacity': False}

        result = update_node_features(x, current_bridges, edge_index, node_type, config)
        assert torch.equal(result, x)


class TestApplyArAction:
    """Test action application to bridge state."""

    def test_normal_action(self):
        """Test applying a normal action."""
        current = torch.tensor([0.0, 1.0, 2.0])
        new_state = apply_ar_action(current, 1, 0)  # Add 1 to edge 0

        expected = torch.tensor([1.0, 1.0, 2.0])
        assert torch.equal(new_state, expected)

    def test_max_bridges_limit(self):
        """Test that actions don't exceed max bridges."""
        current = torch.tensor([1.0, 2.0])
        new_state = apply_ar_action(current, 2, 0)  # Try to add 2 to edge with 1

        expected = torch.tensor([2.0, 2.0])  # Clamped to 2
        assert torch.equal(new_state, expected)

    def test_zero_action(self):
        """Test applying zero action (no change)."""
        current = torch.tensor([0.0, 1.0])
        new_state = apply_ar_action(current, 0, 1)

        assert torch.equal(new_state, current)


class TestSelectArAction:
    """Test action selection from model outputs."""

    def test_regression_head_selection(self):
        """Test action selection with regression head."""
        # Mock output: edge 0 wants ~1.0, edge 1 wants ~0.3, edge 2 locked
        output = torch.tensor([1.0, 0.3, 0.8])
        current_bridges = torch.tensor([0.0, 0.0, 2.0])  # Edge 2 is locked
        edge_mask = torch.tensor([True, True, True])

        edge_idx, confidence = select_ar_action(
            output, current_bridges, edge_mask, head_type='regression'
        )

        # Should select edge 0 (action=1, highest confidence)
        assert edge_idx == 0
        assert confidence > 0.5  # Should be confident

    def test_conditional_head_selection(self):
        """Test action selection with conditional head."""
        # Mock output: edge 0 strongly wants action 1, edge 1 wants action 0
        output = torch.tensor([
            [0.9, 0.2],  # p_ge1=0.9, p_ge2=0.2 -> action 1
            [0.1, 0.8],  # p_ge1=0.1, p_ge2=0.8 -> action 0
        ])
        current_bridges = torch.tensor([0.0, 0.0])
        edge_mask = torch.tensor([True, True])

        edge_idx, confidence = select_ar_action(
            output, current_bridges, edge_mask, head_type='conditional'
        )

        # Should select edge 0 (action 1)
        assert edge_idx == 0

    def test_no_valid_actions(self):
        """Test behavior when no valid actions available."""
        # All edges locked or masked
        output = torch.tensor([1.0, 2.0])
        current_bridges = torch.tensor([2.0, 2.0])  # Both locked
        edge_mask = torch.tensor([True, True])

        edge_idx, confidence = select_ar_action(
            output, current_bridges, edge_mask, head_type='regression'
        )

        # Should return invalid action
        assert edge_idx == -1
        assert confidence == 0.0

    def test_skip_no_op_actions(self):
        """Test that No-Op actions (0) are ignored."""
        # Edge 0 wants action 0 (No-Op), edge 1 wants action 1
        output = torch.tensor([0.3, 1.0])  # Both predict action 1, but 0.3 -> action 0
        current_bridges = torch.tensor([0.0, 0.0])
        edge_mask = torch.tensor([True, True])

        edge_idx, confidence = select_ar_action(
            output, current_bridges, edge_mask, head_type='regression'
        )

        # Should select edge 1 (action 1), skip edge 0 (No-Op)
        assert edge_idx == 1
