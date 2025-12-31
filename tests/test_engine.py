import torch
import unittest
from src.engine import compute_edge_dim, create_model

class TestEngine(unittest.TestCase):
    def test_compute_edge_dim(self):
        # Base features only
        self.assertEqual(compute_edge_dim({}), 3)
        
        # All features
        config = {
            'use_conflict_edges': True,
            'use_meta_mesh': True,
            'use_meta_row_col_edges': True,
            'use_edge_labels_as_features': True
        }
        # 3 (base) + 1 (conflict) + 1 (mesh) + 1 (row_col) + 2 (labels) = 8
        self.assertEqual(compute_edge_dim(config), 8)

    def test_create_model_gcn(self):
        config = {
            'type': 'gcn',
            'node_embedding_dim': 16,
            'hidden_channels': 32,
            'num_layers': 2,
            'use_capacity': True,
            'use_structural_degree': True,
            'use_global_meta_node': True
        }
        device = torch.device('cpu')
        model = create_model(config, device)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertTrue(model.use_capacity)

    def test_create_model_transformer(self):
        config = {
            'type': 'transformer',
            'node_embedding_dim': 16,
            'hidden_channels': 32,
            'num_layers': 2,
            'use_verification_head': True,
            'use_global_meta_node': True
        }
        device = torch.device('cpu')
        model = create_model(config, device)
        self.assertTrue(model.use_verification_head)
        self.assertTrue(hasattr(model, 'verify_mlp'))

if __name__ == '__main__':
    unittest.main()
