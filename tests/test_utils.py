import unittest

import torch
from torch_geometric.data import Batch, Data

from hashi_puzzle_solver.utils.common import custom_collate_with_conflicts, flatten_config


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_flatten_config(self) -> None:
        """Test flatten_config function."""
        config = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3,
                },
            },
        }
        flat = flatten_config(config)
        assert flat["a"] == 1
        assert flat["b.c"] == 2
        assert flat["b.d.e"] == 3

    def test_custom_collate_with_conflicts(self) -> None:
        """Test custom collate function with conflicts."""
        # Create two small graphs with conflicts
        # Graph 0: 2 edges, conflict between 0 and 1
        data0 = Data(
            x=torch.zeros((3, 1)),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_conflicts=[(0, 1)],
        )
        # Graph 1: 3 edges, conflict between 0 and 2
        data1 = Data(
            x=torch.zeros((4, 1)),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            edge_conflicts=[(0, 2)],
        )

        batch = custom_collate_with_conflicts([data0, data1])

        # Total edges = 2 + 3 = 5
        assert batch.edge_index.size(1) == 5

        # Conflict in graph 0 stays (0, 1)
        # Conflict in graph 1 (0, 2) should be offset by graph 0's edge count (2)
        # Result should be [[0, 1], [2, 4]]
        assert hasattr(batch, "edge_conflict_index")
        conflicts = batch.edge_conflict_index
        assert conflicts.size(1) == 2
        assert torch.equal(conflicts, torch.tensor([[0, 2], [1, 4]], dtype=torch.long))

        # Check slicing support
        # Slice to just the second graph using a tensor of indices
        indices = torch.tensor([1], dtype=torch.long)
        # In PyG, to get a Batch back from slicing, we use index_select
        # but in some versions it might still return a list of Data
        sliced_list = batch.index_select(indices)
        if isinstance(sliced_list, list):
            sliced = Batch.from_data_list(sliced_list)
        else:
            sliced = sliced_list

        # Verify it's still a Batch-like object or at least has the attribute
        assert hasattr(sliced, "edge_conflict_index")
        assert sliced.edge_conflict_index.size(1) == 1
        # Should be re-offset back to [[0], [2]]
        assert torch.equal(
            sliced.edge_conflict_index, torch.tensor([[0], [2]], dtype=torch.long)
        )


if __name__ == "__main__":
    unittest.main()
