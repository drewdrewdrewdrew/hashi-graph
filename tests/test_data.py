import unittest
import torch
from unittest.mock import patch
from torch_geometric.data import Data
from src.data import MakeBidirectional, HashiDatasetCache

class TestData(unittest.TestCase):
    def test_make_bidirectional(self):
        # 1 edge: 0 -> 1
        # inv_dx=0.5, inv_dy=0.0
        edge_index = torch.tensor([[0], [1]])
        edge_attr = torch.tensor([[0.5, 0.0, 0.0]]) # [inv_dx, inv_dy, is_meta]
        y = torch.tensor([1])
        edge_mask = torch.tensor([True])
        
        data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y, edge_mask=edge_mask)
        
        transform = MakeBidirectional()
        bi_data = transform(data)
        
        # Should have 2 edges now: 0->1 and 1->0
        self.assertEqual(bi_data.edge_index.size(1), 2)
        self.assertEqual(bi_data.edge_index[0, 1], 1)
        self.assertEqual(bi_data.edge_index[1, 1], 0)
        
        # Edge attrs: reverse edge should have negated inv_dx and inv_dy
        self.assertEqual(bi_data.edge_attr[1, 0], -0.5)
        self.assertEqual(bi_data.edge_attr[1, 1], 0.0)
        
        # Labels and masks should be duplicated
        self.assertEqual(bi_data.y.size(0), 2)
        self.assertTrue(bi_data.edge_mask[1])

    def test_dataset_cache_key_stability(self):
        config1 = {
            'data': {'root_dir': 'dataset/', 'limit': 100},
            'model': {'use_capacity': True, 'use_structural_degree': True}
        }
        config2 = {
            'data': {'root_dir': 'dataset/', 'limit': 100},
            'model': {'use_structural_degree': True, 'use_capacity': True} # Order swapped
        }
        
        hash1 = HashiDatasetCache._config_hash(config1, 'train')
        hash2 = HashiDatasetCache._config_hash(config2, 'train')
        
        self.assertEqual(hash1, hash2)

    def test_grid_stretch(self):
        from src.data import GridStretch
        # 3 nodes in a row: (0,0), (1,0), (2,0)
        pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        edge_index = torch.tensor([[0, 1], [1, 2]])
        # edge_attr: [inv_dx, inv_dy, is_meta]
        edge_attr = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        
        data = Data(pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        
        # Prob=1.0, max_gap=2
        # We try multiple times or just mock axis to be 0
        stretch = GridStretch(prob=1.0, max_gap=2)
        
        # Mock randint to ensure axis 0, split 0, gap 1
        with patch('torch.randint') as mock_randint:
            # First call: axis (0 or 1)
            # Second call: split_idx (0 to len-2)
            # Third call: gap_size (1 to max_gap)
            mock_randint.side_effect = [
                torch.tensor([0]), # axis 0
                torch.tensor([0]), # split_idx 0
                torch.tensor([1])  # gap_size 1
            ]
            stretched_data = stretch(data)
            
        # Verify that positions moved (nodes 1 and 2 should move by +1 in x)
        self.assertEqual(stretched_data.pos[1, 0], 2.0)
        self.assertEqual(stretched_data.pos[2, 0], 3.0)
        
        # Verify that inv_dx updated
        # New distance node 0-1 is 2.0 -> inv_dx = 0.5
        self.assertAlmostEqual(stretched_data.edge_attr[0, 0].item(), 0.5, places=5)
        # New distance node 1-2 is 1.0 -> inv_dx = 1.0
        self.assertAlmostEqual(stretched_data.edge_attr[1, 0].item(), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()

