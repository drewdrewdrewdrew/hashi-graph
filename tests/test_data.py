import unittest
import torch
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

if __name__ == '__main__':
    unittest.main()

