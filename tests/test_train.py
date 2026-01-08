import unittest
import torch
import numpy as np
from torch_geometric.data import Data
from src.engine import get_masking_rate, apply_edge_label_masking

class TestTrainLogic(unittest.TestCase):
    def test_get_masking_rate_schedules(self):
        config = {
            'enabled': True,
            'warmup_epochs': 10,
            'cooldown_epochs': 10,
            'start_rate': 0.0,
            'end_rate': 1.0,
            'schedule': 'linear'
        }
        total_epochs = 100
        
        # Warmup
        self.assertEqual(get_masking_rate(5, config, total_epochs), 0.0)
        # Cooldown
        self.assertEqual(get_masking_rate(95, config, total_epochs), 1.0)
        # Linear Midpoint: (50-10)/(100-10-10) = 40/80 = 0.5
        self.assertEqual(get_masking_rate(50, config, total_epochs), 0.5)

    def test_apply_edge_label_masking(self):
        # 2 nodes, 1 edge
        # edge_attr: [inv_dx, inv_dy, is_meta, bridge_label, is_labeled]
        edge_attr = torch.tensor([[0.5, 0.0, 0.0, 1.0, 1.0]])
        x = torch.tensor([[5.0, 2.0, 5.0], [5.0, 2.0, 5.0]]) # [cap, struct, unused]
        edge_index = torch.tensor([[0], [1]])
        edge_mask = torch.tensor([True])
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_mask=edge_mask)
        config = {
            'model': {
                'use_capacity': True,
                'use_structural_degree': True,
                'use_unused_capacity': True
            }
        }
        
        # Apply 100% masking
        masked_data = apply_edge_label_masking(data, 1.0, torch.device('cpu'), config)
        
        # Features 3 and 4 (label and is_labeled) should be 0
        self.assertEqual(masked_data.edge_attr[0, 3], 0.0)
        self.assertEqual(masked_data.edge_attr[0, 4], 0.0)
        # Unused capacity (idx 2) should be incremented by the masked bridge label (1.0)
        self.assertEqual(masked_data.x[0, 2], 6.0)

if __name__ == '__main__':
    unittest.main()



