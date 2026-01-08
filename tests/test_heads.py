"""
Unit tests for AR action heads.
"""
import torch
import pytest
from hashi_puzzle_solver.models.heads import RegressionActionHead, ConditionalActionHead


class TestRegressionActionHead:
    """Test regression head functionality."""

    def test_output_range(self):
        """Test that outputs are constrained to [0, 2] range."""
        head = RegressionActionHead(input_dim=64)
        x = torch.randn(10, 64)

        output = head(x)

        # Should be in [0, 2] range (with small buffer)
        assert output.min() >= -0.1, f"Min output {output.min()} below -0.1"
        assert output.max() <= 2.1, f"Max output {output.max()} above 2.1"

    def test_predict_action_bins(self):
        """Test that predict_action correctly bins outputs."""
        head = RegressionActionHead(input_dim=64)

        # Test boundary values
        test_values = [0.0, 0.5, 0.7, 1.0, 1.2, 1.4, 2.0]
        expected_actions = [0, 0, 1, 1, 1, 2, 2]

        for val, expected in zip(test_values, expected_actions):
            output = torch.tensor([val])
            action, confidence = head.predict_action(output)
            assert action == expected, f"Value {val} should map to action {expected}, got {action}"

    def test_predict_action_confidence(self):
        """Test that confidence scores make sense."""
        head = RegressionActionHead(input_dim=64)

        # Test at bin centers (should have high confidence)
        center_values = [0.333, 1.0, 1.666]
        for val in center_values:
            output = torch.tensor([val])
            action, confidence = head.predict_action(output)
            assert confidence > 0.5, f"Center value {val} should have high confidence, got {confidence}"

        # Test at bin boundaries (should have lower confidence)
        boundary_values = [0.666, 1.333]
        for val in boundary_values:
            output = torch.tensor([val])
            action, confidence = head.predict_action(output)
            assert confidence < 0.8, f"Boundary value {val} should have lower confidence, got {confidence}"


class TestConditionalActionHead:
    """Test conditional head functionality."""

    def test_output_shape(self):
        """Test that outputs have correct shape."""
        head = ConditionalActionHead(input_dim=64)
        x = torch.randn(10, 64)

        output = head(x)

        assert output.shape == (10, 2), f"Expected shape (10, 2), got {output.shape}"

    def test_output_range(self):
        """Test that outputs are in [0, 1] range."""
        head = ConditionalActionHead(input_dim=64)
        x = torch.randn(10, 64)

        output = head(x)

        assert output.min() >= 0.0, f"Min output {output.min()} below 0"
        assert output.max() <= 1.0, f"Max output {output.max()} above 1"

    def test_predict_action_probabilities(self):
        """Test that action probabilities sum correctly."""
        head = ConditionalActionHead(input_dim=64)

        # Create specific test case
        # p_ge1 = 0.8, p_ge2 = 0.6
        # Should give: p0=0.2, p1=0.8*0.4=0.32, p2=0.8*0.6=0.48
        # So action should be 2 with confidence 0.48
        test_output = torch.tensor([0.8, 0.6])
        action, confidence = head.predict_action(test_output)

        assert action == 2, f"Expected action 2, got {action}"
        assert abs(confidence - 0.48) < 0.01, f"Expected confidence ~0.48, got {confidence}"

    def test_predict_action_edge_cases(self):
        """Test edge cases for action prediction."""
        head = ConditionalActionHead(input_dim=64)

        # Case 1: p_ge1=0 (no bridges ever)
        test_output = torch.tensor([0.0, 0.5])
        action, confidence = head.predict_action(test_output)
        assert action == 0, f"Expected action 0 when p_ge1=0, got {action}"

        # Case 2: p_ge1=1, p_ge2=0 (exactly 1 bridge)
        test_output = torch.tensor([1.0, 0.0])
        action, confidence = head.predict_action(test_output)
        assert action == 1, f"Expected action 1 when p_ge1=1, p_ge2=0, got {action}"

        # Case 3: p_ge1=1, p_ge2=1 (exactly 2 bridges)
        test_output = torch.tensor([1.0, 1.0])
        action, confidence = head.predict_action(test_output)
        assert action == 2, f"Expected action 2 when both probs=1, got {action}"
