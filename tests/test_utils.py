import unittest
import torch
from torch_geometric.data import Data
from src.utils import custom_collate_with_conflicts, flatten_config

class TestUtils(unittest.TestCase):
    def test_flatten_config(self):
        config = {
            'a': 1,
            'b': {
                'c': 2,
                'd': {
                    'e': 3
                }
            }
        }
        flat = flatten_config(config)
        self.assertEqual(flat['a'], 1)
        self.assertEqual(flat['b.c'], 2)
        self.assertEqual(flat['b.d.e'], 3)

    def test_custom_collate_with_conflicts(self):
        # Create two small graphs with conflicts
        # Graph 0: 2 edges, conflict between 0 and 1
        data0 = Data(
            x=torch.zeros((3, 1)),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_conflicts=[(0, 1)]
        )
        # Graph 1: 3 edges, conflict between 0 and 2
        data1 = Data(
            x=torch.zeros((4, 1)),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            edge_conflicts=[(0, 2)]
        )
        
        batch = custom_collate_with_conflicts([data0, data1])
        
        # Total edges = 2 + 3 = 5
        self.assertEqual(batch.edge_index.size(1), 5)
        
        # Conflict in graph 0 stays (0, 1)
        # Conflict in graph 1 (0, 2) should be offset by graph 0's edge count (2)
        # Result should be [[0, 1], [2, 4]]
        conflicts = batch.edge_conflicts
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(conflicts[0], [0, 1])
        self.assertEqual(conflicts[1], [2, 4])

if __name__ == '__main__':
    unittest.main()

