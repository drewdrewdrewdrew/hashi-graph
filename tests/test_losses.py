import unittest
import torch
from src.losses import compute_degree_violation_loss, compute_crossing_loss, compute_verification_loss

class TestLosses(unittest.TestCase):
    def test_degree_violation_loss(self):
        # 3 nodes, 4 edges (bidirectional): 0-1, 1-0, 0-2, 2-0
        # Node 0 capacity: 3. Edges from 0: 0->1 (1), 0->2 (2). Sum = 3.
        # Node 1 capacity: 1. Edge from 1: 1->0 (1). Sum = 1.
        # Node 2 capacity: 2. Edge from 2: 2->0 (2). Sum = 2.
        logits = torch.tensor([
            [0.0, 10.0, -10.0], # 0->1 (label 1)
            [0.0, 10.0, -10.0], # 1->0 (label 1)
            [-10.0, 0.0, 10.0], # 0->2 (label 2)
            [-10.0, 0.0, 10.0]  # 2->0 (label 2)
        ])
        edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
        node_capacities = torch.tensor([3, 1, 2])
        edge_mask = torch.tensor([True, True, True, True])
        
        loss = compute_degree_violation_loss(logits, edge_index, node_capacities, edge_mask)
        self.assertLess(loss.item(), 0.1)

    def test_crossing_loss(self):
        # 2 conflicting edges
        # Edge 0 high prob active, Edge 1 high prob active.
        logits = torch.tensor([
            [-10.0, 10.0, -10.0], # Active (label 1)
            [-10.0, -10.0, 10.0]  # Active (label 2)
        ])
        edge_conflicts = [(0, 1)]
        edge_mask = torch.tensor([True, True])
        
        loss = compute_crossing_loss(logits, edge_conflicts, edge_mask)
        # Multiplicative: ~1.0 * ~1.0 = 1.0
        self.assertGreater(loss.item(), 0.9)

    def test_verification_loss(self):
        # 1 puzzle, perfect prediction
        # verify_logits [1, 1], edge_logits [num_edges, 3]
        verify_logits = torch.tensor([[10.0]]) # Predicting perfect
        edge_logits = torch.tensor([[0.0, 10.0, 0.0]]) # Prediction: 1
        targets = torch.tensor([1]) # Correct
        edge_mask = torch.tensor([True])
        edge_batch = torch.tensor([0])
        
        loss, balanced_acc, recall_pos, recall_neg = compute_verification_loss(
            verify_logits, edge_logits, targets, edge_mask, edge_batch
        )
        # Prediction matches reality, so verify_target is 1.0. 
        # Logit 10.0 ~ prob 1.0. Loss ~ 0.
        self.assertLess(loss.item(), 0.1)
        self.assertEqual(balanced_acc.item(), 1.0)

if __name__ == '__main__':
    unittest.main()

