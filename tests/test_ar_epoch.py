"""Unit tests for AR (Incremental) training functionality."""


import pytest
import torch
from torch_geometric.data import Batch, Data

from hashi_puzzle_solver.engine import initialize_random_bridge_state, run_ar_epoch
from hashi_puzzle_solver.train_utils import (
    apply_ar_action,
    select_ar_action,
    update_node_features,
)


class MockModel(torch.nn.Module):
    """Mock model for testing AR epoch functionality."""

    def __init__(self, output_value: float = 1.0) -> None:
        super().__init__()
        self.output_value = output_value
        # Add a dummy parameter so next(model.parameters()) works
        self.dummy_param = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        _x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,  # noqa: ARG002
        _batch: torch.Tensor | None = None,
        _node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return constant output for testing."""
        num_edges = edge_index.size(1)
        return torch.full((num_edges,), self.output_value)


@pytest.fixture
def simple_puzzle_data() -> Data:
    """Create a simple 2x2 puzzle for testing."""
    # Simple puzzle: 4 nodes, 4 edges
    edge_index = torch.tensor([
        [0, 1, 2, 3],  # source nodes
        [1, 2, 3, 0],  # target nodes
    ])

    # Node features: [capacity, structural_degree, unused_capacity]
    x = torch.tensor([
        [1.0, 1.0, 1.0],  # node 0
        [1.0, 1.0, 1.0],  # node 1
        [1.0, 1.0, 1.0],  # node 2
        [0.0, 0.0, 0.0],  # node 3 (meta node)
    ])

    # Edge labels (ground truth)
    y = torch.tensor([1, 1, 1, 0])  # 3 bridges of 1, 1 bridge of 0

    # Node types (islands are 1-8)
    node_type = torch.tensor([1, 2, 3, 9])  # 9 = meta

    # Edge mask (which edges are puzzle edges vs meta)
    edge_mask = torch.tensor([True, True, True, False])

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        node_type=node_type,
        edge_mask=edge_mask,
    )


@pytest.fixture
def mock_model_config() -> dict:
    """Mock model configuration."""
    return {
        "use_capacity": True,
        "use_structural_degree": True,
        "use_unused_capacity": True,
    }


def test_initialize_random_bridge_state(simple_puzzle_data: Data) -> None:
    """Test random bridge state initialization."""
    final_bridges = simple_puzzle_data.y
    edge_mask = simple_puzzle_data.edge_mask

    # Test multiple random initializations
    for _ in range(10):
        current_bridges = initialize_random_bridge_state(final_bridges, edge_mask)

        # Check shape
        assert current_bridges.shape == final_bridges.shape

        # Check that initialized bridges don't exceed final bridges
        assert torch.all(current_bridges <= final_bridges)

        # Check that bridges are integers (0, 1, 2)
        assert torch.all(
            (current_bridges == 0) | (current_bridges == 1) | (current_bridges == 2),
        )

        # Check that meta edges (non-masked) are 0
        assert current_bridges[~edge_mask].sum() == 0


def test_update_node_features(
    simple_puzzle_data: Data, mock_model_config: dict
) -> None:
    """Test node feature updates based on bridge state."""
    current_bridges = torch.tensor([1, 0, 0, 0])  # One bridge placed on edge 0

    updated_x = update_node_features(
        simple_puzzle_data.x.clone(),
        current_bridges,
        simple_puzzle_data.edge_index,
        simple_puzzle_data.node_type,
        mock_model_config,
    )

    # Node 0 should have unused_capacity = 1.0 - 1 = 0.0
    # Node 1 should have unused_capacity = 1.0 - 1 = 0.0
    # Node 2 should have unused_capacity = 1.0 (unchanged)
    # Node 3 should have unused_capacity = 0.0 (meta node)
    expected_unused_capacity = torch.tensor([0.0, 0.0, 1.0, 0.0])

    unused_capacity_idx = (
        2  # unused_capacity is at index 2 (after capacity and structural_degree)
    )
    assert torch.allclose(updated_x[:, unused_capacity_idx], expected_unused_capacity)


def test_apply_ar_action() -> None:
    """Test applying AR actions to bridge state."""
    current_bridges = torch.tensor([0, 1, 0, 0])

    # Apply action +1 to edge 0
    new_bridges = apply_ar_action(current_bridges, 1, 0)
    expected = torch.tensor([1, 1, 0, 0])
    assert torch.equal(new_bridges, expected)

    # Apply action +2 to edge 2 (should be clamped to 2)
    new_bridges = apply_ar_action(new_bridges, 2, 2)
    expected = torch.tensor([1, 1, 2, 0])
    assert torch.equal(new_bridges, expected)

    # Apply another +1 to edge 2 (should stay at 2)
    new_bridges = apply_ar_action(new_bridges, 1, 2)
    expected = torch.tensor([1, 1, 2, 0])
    assert torch.equal(new_bridges, expected)


def test_select_ar_action_regression() -> None:
    """Test action selection for regression head."""
    # Mock model output (regression values)
    output = torch.tensor([0.5, 1.5, 2.1, 0.1])  # Actions: +1, +2, +2, +0
    current_bridges = torch.tensor([0, 0, 1, 0])
    edge_mask = torch.tensor([True, True, True, False])

    edge_idx, _confidence = select_ar_action(
        output, current_bridges, edge_mask, "regression",
    )

    # Should select edge 1 with action +2 (highest confidence constructive action)
    # Edge 2 has current_bridges=1 and output=2.1 -> action +2, but 1+2=3>2, so invalid
    # Edge 0: action +1, valid
    # Edge 1: action +2, valid
    # So should pick edge 1 with highest confidence
    assert edge_idx == 1


def test_select_ar_action_conditional() -> None:
    """Test action selection for conditional head."""
    # Mock conditional output: [P(≥1), P(=2)]
    output = torch.tensor([
        [0.8, 0.3],  # Edge 0: P(≥1)=0.8, P(=2)=0.3 -> action +1
        [0.9, 0.7],  # Edge 1: P(≥1)=0.9, P(=2)=0.7 -> action +2
        [0.2, 0.1],  # Edge 2: P(≥1)=0.2, P(=2)=0.1 -> action +0
        [0.5, 0.4],  # Edge 3: masked out
    ])
    current_bridges = torch.tensor([0, 0, 0, 0])
    edge_mask = torch.tensor([True, True, True, False])

    edge_idx, _confidence = select_ar_action(
        output, current_bridges, edge_mask, "conditional",
    )

    # Should select edge 1 with action +2 (highest confidence)
    assert edge_idx == 1


def test_run_ar_epoch_basic(simple_puzzle_data: Data, mock_model_config: dict) -> None:
    """Test basic AR epoch execution."""
    model = MockModel(output_value=1.0)

    # Create a simple dataloader
    batch = Batch.from_data_list([simple_puzzle_data])
    loader = [batch]  # Mock loader as list for iteration

    # Run AR epoch
    metrics = run_ar_epoch(
        model=model,
        loader=loader,
        ar_steps=2,
        head_type="regression",
        training=False,
        model_config=mock_model_config,
    )

    # Check that we got metrics back
    assert isinstance(metrics.loss, float)
    assert metrics.loss >= 0.0


def test_run_ar_epoch_with_random_start(
    simple_puzzle_data: Data, mock_model_config: dict
) -> None:
    """Test AR epoch with random starting states."""
    model = MockModel(output_value=0.5)

    # Create multiple batches to test random initialization
    batches = [Batch.from_data_list([simple_puzzle_data]) for _ in range(3)]
    loader = batches

    metrics = run_ar_epoch(
        model=model,
        loader=loader,
        ar_steps=3,
        head_type="regression",
        training=False,
        model_config=mock_model_config,
    )

    # Should complete without errors
    assert isinstance(metrics.loss, float)


if __name__ == "__main__":
    pytest.main([__file__])
