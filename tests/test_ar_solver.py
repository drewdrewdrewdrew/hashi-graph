"""
Unit tests for AR (Incremental) solver functionality.
"""
import torch
import pytest
from torch_geometric.data import Data

from hashi_puzzle_solver.ar_solver import solve_incremental
from hashi_puzzle_solver.train_utils import update_node_features


class MockARModel(torch.nn.Module):
    """Mock AR model for testing solver."""

    def __init__(self, action_sequence):
        """
        Mock model that returns predetermined actions.

        Args:
            action_sequence: List of (edge_idx, action) tuples to return in order
        """
        super().__init__()
        self.action_sequence = action_sequence
        self.call_count = 0

    def forward(self, x, edge_index, edge_attr=None, batch=None, node_type=None):
        """Return mock outputs based on predetermined sequence."""
        if self.call_count < len(self.action_sequence):
            edge_idx, action = self.action_sequence[self.call_count]
            num_edges = edge_index.size(1)
            # Create output that will predict the desired action for edge_idx
            output = torch.zeros(num_edges)
            if action == 1:
                output[edge_idx] = 1.0  # Regression: 1.0 -> action +1
            elif action == 2:
                output[edge_idx] = 2.0  # Regression: 2.0 -> action +2
            self.call_count += 1
            return output
        else:
            # Return zeros when no more actions
            return torch.zeros(edge_index.size(1))


@pytest.fixture
def simple_solveable_puzzle():
    """Create a simple solvable puzzle for testing."""
    # 2x2 grid puzzle that can be solved in 2 steps
    edge_index = torch.tensor([
        [0, 1, 2, 3],  # source nodes
        [1, 2, 3, 0]   # target nodes
    ])

    # Node features: [capacity, structural_degree, unused_capacity]
    x = torch.tensor([
        [1.0, 1.0, 1.0],  # node 0: needs 1 bridge
        [1.0, 1.0, 1.0],  # node 1: needs 1 bridge
        [0.0, 0.0, 0.0],  # node 2: meta node
        [0.0, 0.0, 0.0],  # node 3: meta node
    ])

    # Edge labels (ground truth) - solution: bridge between 0-1
    y = torch.tensor([1, 0, 0, 0])

    # Node types (also serve as capacities: 1=capacity 1, 2=capacity 2, 9=meta)
    node_type = torch.tensor([1, 1, 9, 9])  # Both nodes need 1 bridge total

    # Edge mask
    edge_mask = torch.tensor([True, False, False, False])

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        node_type=node_type,
        edge_mask=edge_mask
    )

    return data


@pytest.fixture
def model_config():
    """Standard model configuration."""
    return {
        'use_capacity': True,
        'use_structural_degree': True,
        'use_unused_capacity': True,
    }


def test_solve_incremental_simple_puzzle(simple_solveable_puzzle, model_config):
    """Test incremental solving of a simple puzzle."""
    # Model that places 1 bridge on edge 0
    action_sequence = [(0, 1)]
    model = MockARModel(action_sequence)

    solved, steps_taken = solve_incremental(
        model=model,
        initial_state=simple_solveable_puzzle,
        max_steps=5,
        head_type='regression',
        model_config=model_config
    )

    assert solved == True
    assert steps_taken == 1


def test_solve_incremental_timeout(simple_solveable_puzzle, model_config):
    """Test solver timeout when no progress is made."""
    # Model that always returns invalid actions
    model = MockARModel([])  # Empty sequence = always return zeros

    solved, steps_taken = solve_incremental(
        model=model,
        initial_state=simple_solveable_puzzle,
        max_steps=3,
        head_type='regression',
        model_config=model_config
    )

    assert solved == False
    assert steps_taken == 3  # Should reach max_steps


def test_solve_incremental_complex_puzzle(model_config):
    """Test solving a more complex puzzle requiring multiple steps."""
    # Create a puzzle requiring 2 steps
    edge_index = torch.tensor([
        [0, 1, 2, 0, 1],  # source nodes
        [1, 2, 0, 2, 0]   # target nodes
    ])

    x = torch.tensor([
        [2.0, 2.0, 2.0],  # node 0: needs 2 bridges total
        [1.0, 1.0, 1.0],  # node 1: needs 1 bridge
        [1.0, 1.0, 1.0],  # node 2: needs 1 bridge
    ])

    # Solution: edge 0-1: 1 bridge, edge 2-0: 1 bridge (node 0 gets 2 bridges total)
    y = torch.tensor([1, 0, 1, 0, 0])

    node_type = torch.tensor([2, 1, 1])  # node 0 needs 2 bridges, nodes 1&2 need 1 each
    edge_mask = torch.tensor([True, False, True, False, False])

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        node_type=node_type,
        edge_mask=edge_mask
    )

    # Model that first places 1 bridge on edge 0, then 1 bridge on edge 2
    action_sequence = [(0, 1), (2, 1)]
    model = MockARModel(action_sequence)

    solved, steps_taken = solve_incremental(
        model=model,
        initial_state=data,
        max_steps=5,
        head_type='regression',
        model_config=model_config
    )

    assert solved == True
    assert steps_taken == 2


def test_solve_incremental_invalid_actions(model_config):
    """Test solver handles invalid actions gracefully."""
    # Create puzzle where some actions would be invalid
    edge_index = torch.tensor([[0, 1], [1, 0]])
    x = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    y = torch.tensor([1, 0])  # Only edge 0 should have a bridge
    node_type = torch.tensor([1, 1])  # Both nodes need 1 bridge total
    edge_mask = torch.tensor([True, True])

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        node_type=node_type,
        edge_mask=edge_mask
    )

    # Model tries to place bridge on edge 1 first (invalid), then on edge 0
    action_sequence = [(1, 1), (0, 1)]  # First action invalid, second valid
    model = MockARModel(action_sequence)

    solved, steps_taken = solve_incremental(
        model=model,
        initial_state=data,
        max_steps=5,
        head_type='regression',
        model_config=model_config
    )

    assert solved == True
    assert steps_taken == 1  # Only the valid action counts


def test_solve_incremental_conditional_head(simple_solveable_puzzle, model_config):
    """Test solver with conditional action head."""
    class MockConditionalModel(MockARModel):
        def forward(self, x, edge_index, edge_attr=None, batch=None, node_type=None):
            """Return conditional head format: [num_edges, 2]"""
            if self.call_count < len(self.action_sequence):
                edge_idx, action = self.action_sequence[self.call_count]
                num_edges = edge_index.size(1)
                output = torch.zeros(num_edges, 2)  # [P(≥1), P(=2)]

                if action == 1:
                    output[edge_idx] = torch.tensor([0.9, 0.1])  # P(≥1)=0.9, P(=2)=0.1
                elif action == 2:
                    output[edge_idx] = torch.tensor([0.8, 0.8])  # P(≥1)=0.8, P(=2)=0.8

                self.call_count += 1
                return output
            else:
                return torch.zeros(edge_index.size(1), 2)

    action_sequence = [(0, 1)]
    model = MockConditionalModel(action_sequence)

    solved, steps_taken = solve_incremental(
        model=model,
        initial_state=simple_solveable_puzzle,
        max_steps=5,
        head_type='conditional',
        model_config=model_config
    )

    assert solved == True
    assert steps_taken == 1


if __name__ == "__main__":
    pytest.main([__file__])
