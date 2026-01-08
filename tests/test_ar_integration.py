"""Integration tests for AR system components."""

import torch

from hashi_puzzle_solver.losses import compute_ar_loss
from hashi_puzzle_solver.models import TransformerEdgeClassifier
from hashi_puzzle_solver.train_utils import select_ar_action, update_node_features


def test_ar_model_instantiation() -> None:
    """Test that AR models can be created and run forward passes."""
    # Test regression head
    model_reg = TransformerEdgeClassifier(
        node_embedding_dim=64,
        hidden_channels=128,
        num_layers=2,
        head_type="regression",
    )

    # Test conditional head
    model_cond = TransformerEdgeClassifier(
        node_embedding_dim=64,
        hidden_channels=128,
        num_layers=2,
        head_type="conditional",
    )

    # Create dummy input with proper feature format
    # Features: [capacity, structural_degree, unused_capacity, conflict_status] # noqa: ERA001, E501
    x = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],  # node 0: capacity 1, no conflict
            [1.0, 1.0, 1.0, 0.0],  # node 1: capacity 1, no conflict
            [1.0, 1.0, 1.0, 0.0],  # node 2: capacity 1, no conflict
            [1.0, 1.0, 1.0, 0.0],  # node 3: capacity 1, no conflict
            [0.0, 0.0, 0.0, 0.0],  # node 4: meta node
            [0.0, 0.0, 0.0, 0.0],  # node 5: meta node
            [0.0, 0.0, 0.0, 0.0],  # node 6: meta node
            [0.0, 0.0, 0.0, 0.0],  # node 7: meta node
            [0.0, 0.0, 0.0, 0.0],  # node 8: meta node
            [0.0, 0.0, 0.0, 0.0],  # node 9: meta node
        ],
        dtype=torch.float,
    )  # 10 nodes, 4 features
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 3 edges

    # Forward passes
    output_reg = model_reg(x, edge_index)
    output_cond = model_cond(x, edge_index)

    # Check shapes
    assert output_reg.shape == (3,)  # Regression: 1 value per edge
    assert output_cond.shape == (3, 2)  # Conditional: 2 values per edge

    print("✓ AR models instantiate and forward pass correctly")


def test_ar_loss_computation() -> None:
    """Test that AR loss functions work correctly."""
    # Create dummy data
    output_reg = torch.tensor([1.2, 0.8, 1.5])  # Regression predictions
    output_cond = torch.tensor([
        [0.9, 0.1],
        [0.3, 0.6],
        [0.8, 0.2],
    ])  # Conditional predictions
    targets = torch.tensor([1.0, 1.0, 2.0])  # Remaining capacity targets
    current_bridges = torch.tensor([0.0, 0.0, 0.0])
    edge_mask = torch.tensor([True, True, True])

    # Test regression loss
    loss_reg = compute_ar_loss(
        output_reg,
        targets,
        current_bridges,
        edge_mask,
        head_type="regression",
        overshoot_penalty=2.0,
    )

    # Test conditional loss
    loss_cond = compute_ar_loss(
        output_cond,
        targets,
        current_bridges,
        edge_mask,
        head_type="conditional",
    )

    # Check that losses are valid scalars
    assert loss_reg.item() > 0
    assert loss_cond.item() > 0

    print("✓ AR loss computation works correctly")


def test_ar_state_updates() -> None:
    """Test that node feature updates work correctly."""
    # Mock data similar to actual puzzles
    x = torch.tensor([
        [1.0, 2.0, 4.0],  # Island 1: capacity=1, unused=4
        [2.0, 1.0, 2.0],  # Island 2: capacity=2, unused=2
        [0.0, 0.0, 0.0],  # Meta node
    ])

    edge_index = torch.tensor([
        [0, 0],  # Edge between islands (0->1), Edge to meta (0->2)
        [1, 2],  #
    ])

    current_bridges = torch.tensor([1.0, 0.0])  # 1 bridge on first edge
    node_type = torch.tensor([1, 2, 9])  # Island types

    config = {
        "use_capacity": True,
        "use_structural_degree": True,
        "use_unused_capacity": True,
    }

    updated_x = update_node_features(x, current_bridges, edge_index, node_type, config)

    # Island 1: original unused=4, used 1 bridge -> remaining 3
    # Island 2: original unused=2, used 1 bridge -> remaining 1
    # Meta nodes: unchanged
    expected = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ])

    assert torch.allclose(updated_x, expected)

    print("✓ AR state updates work correctly")


def test_ar_action_selection() -> None:
    """Test that action selection works correctly."""
    # Test regression head
    output_reg = torch.tensor([1.2, 0.5, 1.8])  # [~1, ~0, ~2]
    current_bridges = torch.tensor([0.0, 0.0, 0.0])
    edge_mask = torch.tensor([True, True, True])

    edge_idx, confidence = select_ar_action(
        output_reg, current_bridges, edge_mask, "regression",
    )

    # Should select edge 2 (action 2) as it's most confident positive action
    assert edge_idx == 2
    assert confidence > 0.5

    # Test conditional head
    output_cond = torch.tensor([
        [0.9, 0.1],  # Strong preference for action 1
        [0.2, 0.7],  # Strong preference for action 2
    ])
    current_bridges = torch.tensor([0.0, 0.0])
    edge_mask = torch.tensor([True, True])

    edge_idx, confidence = select_ar_action(
        output_cond, current_bridges, edge_mask, "conditional",
    )

    # Should select edge 0 (action 1) or edge 1 (action 2), depending on confidence
    assert edge_idx in [0, 1]
    assert confidence > 0.3  # Should have reasonable confidence

    print("✓ AR action selection works correctly")


if __name__ == "__main__":
    test_ar_model_instantiation()
    test_ar_loss_computation()
    test_ar_state_updates()
    test_ar_action_selection()
    print("\n🎉 All AR integration tests passed!")
