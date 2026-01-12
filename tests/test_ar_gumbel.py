
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import pytest

from hashi_puzzle_solver.ar_engine import ARTrainer, ARState

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_verification_head = False
        # A simple parameter to ensure we have something to optimize
        self.weight = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def forward(self, x, edge_index, edge_attr=None, batch=None, node_type=None, return_verification=False):
        num_edges = edge_index.size(1)
        # Return constant logits modified by weight to ensure gradient flow
        # Shape: [num_edges, 3]
        logits = torch.randn(num_edges, 3, device=x.device) * self.weight
        
        if return_verification:
            return logits, torch.randn(x.size(0), 1, device=x.device)
        return logits

def test_ar_gumbel_gradients():
    """Test that gradients flow through the AR rollout using Gumbel-Softmax."""
    # Setup data
    # 2 islands, 1 edge
    x = torch.zeros(2, 5) # 5 features
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    y = torch.tensor([1, 1], dtype=torch.long)
    edge_mask = torch.tensor([True, True], dtype=torch.bool)
    node_type = torch.tensor([1, 1], dtype=torch.long)
    # edge_attr: [dist, direction, bridge_label, is_labeled, etc]
    edge_attr = torch.zeros(2, 5) 

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_mask=edge_mask, node_type=node_type)
    
    # Config
    config = {
        "model": {
            "use_verification_head": False,
            "bridge_label_idx": 2,
            "is_labeled_idx": 3,
            "use_unused_capacity": True,
            "use_capacity": False,
            "use_structural_degree": False,
            "use_component_meta": False,
        },
        "training": {
            "ar_max_steps": 5,
            "gumbel_temperature": 1.0,
            "loss_weights": {"ce": 1.0, "degree": 0.0, "crossing": 0.0, "verify": 0.0},
        }
    }

    device = torch.device("cpu")
    model = MockModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    trainer = ARTrainer(model, config, device)
    
    # Manually set indices for the trainer
    trainer.bridge_label_idx = 2
    trainer.is_labeled_idx = 3

    loader = DataLoader([data], batch_size=1, collate_fn=lambda x: Batch.from_data_list(x))

    # Run epoch
    metrics = trainer.run_epoch(loader, epoch=1, total_epochs=1, optimizer=optimizer, training=True)
    
    assert metrics["loss"] > 0
    assert model.weight.grad is not None
    assert torch.norm(model.weight.grad) > 0
