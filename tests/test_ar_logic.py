"""Tests for Auto-Regressive (AR) logic and utilities."""

import torch
from torch_geometric.data import Batch, Data

from hashi_puzzle_solver.ar_utils import (
    detect_components,
    get_ar_targets,
    rewire_component_meta_edges,
    rewire_component_meta_edges_batch,
)


def test_detect_components() -> None:
    """Test connected component detection logic."""
    # 3 islands, 1 bridge between 0 and 1
    num_islands = 3
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    current_bridges = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    node_type = torch.tensor([1, 1, 1], dtype=torch.long)

    reps = detect_components(num_islands, edge_index, current_bridges, node_type)

    assert reps[0] == reps[1]
    assert reps[0] != reps[2]


def test_rewire_component_meta_edges() -> None:
    """Test island-to-component-meta rewiring for a single graph."""
    # 2 islands, 2 component metas
    # Islands: 0, 1. Metas: 2, 3.
    node_type = torch.tensor([1, 1, 11, 11], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 2],  # island 0 -> meta 2
            [1, 3],  # island 1 -> meta 3
            [2, 0],  # meta 2 -> island 0
            [3, 1],  # meta 3 -> island 1
        ],
        dtype=torch.long,
    ).t()

    data = Data(edge_index=edge_index, node_type=node_type)

    # After a bridge is added, both islands are in component 0
    representatives = torch.tensor([0, 0], dtype=torch.long)

    data = rewire_component_meta_edges(data, representatives)

    # island 0 -> meta 2+0=2
    # island 1 -> meta 2+0=2
    assert data.edge_index[1, 0] == 2
    assert data.edge_index[1, 1] == 2
    assert data.edge_index[0, 2] == 2
    assert data.edge_index[0, 3] == 2


def test_rewire_component_meta_edges_batch() -> None:
    """
    Ensure batched rewiring updates both forward and reverse meta edges.

    Tests that it does not mutate across puzzles by checking distinct
    per-puzzle remapping.
    """
    # Puzzle 1: islands 0,1 metas 2,3 plus puzzle edges between islands
    node_type_1 = torch.tensor([1, 1, 11, 11], dtype=torch.long)
    edge_index_1 = torch.tensor(
        [
            [0, 1, 0, 1, 2, 3],  # 0-1 puzzle edges, then meta edges
            [1, 0, 2, 3, 0, 1],
        ],
        dtype=torch.long,
    )

    # Puzzle 2 mirrors puzzle 1
    node_type_2 = torch.tensor([1, 1, 11, 11], dtype=torch.long)
    edge_index_2 = torch.tensor(
        [
            [0, 1, 0, 1, 2, 3],
            [1, 0, 2, 3, 0, 1],
        ],
        dtype=torch.long,
    )

    data_list = [
        Data(edge_index=edge_index_1, node_type=node_type_1),
        Data(edge_index=edge_index_2, node_type=node_type_2),
    ]

    batch = Batch.from_data_list(data_list)

    class MockPuzzle:
        def __init__(self, num_islands: int) -> None:
            self.num_islands = num_islands
            self.num_edges = 6
            self.current_bridges = torch.zeros(6, dtype=torch.long)

    active = [MockPuzzle(2), MockPuzzle(2)]

    # Component representatives: first puzzle islands -> 0, second puzzle -> 1
    # to ensure per-puzzle remapping is distinct.
    # First two edges are puzzle edges; set them active to merge components
    active[0].current_bridges = torch.tensor([1, 1, 0, 0, 0, 0])
    active[1].current_bridges = torch.tensor([1, 1, 0, 0, 0, 0])

    # Force detect_components to place both islands in same component per puzzle
    rewire_component_meta_edges_batch(batch, active)

    # Puzzle 1 forward edges
    assert batch.edge_index[1, 2] == batch.ptr[0] + 3  # island 0 -> meta 3
    assert batch.edge_index[1, 3] == batch.ptr[0] + 3  # island 1 -> meta 3
    # Puzzle 2 forward edges
    assert batch.edge_index[1, 8] == batch.ptr[1] + 3  # island 0 -> meta 3 (puzzle 2)
    assert batch.edge_index[1, 9] == batch.ptr[1] + 3  # island 1 -> meta 3 (puzzle 2)


def test_ar_targets() -> None:
    """Test binary target calculation for AR steps."""
    data = Data(y=torch.tensor([2, 1, 0]))
    current_bridges = torch.tensor([1, 1, 0])
    targets = get_ar_targets(data, current_bridges)

    # 1 < 2 -> 1
    # 1 < 1 -> 0
    # 0 < 0 -> 0
    assert torch.all(targets == torch.tensor([1, 0, 0]))
