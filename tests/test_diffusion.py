import torch
import pytest
from torch_geometric.data import Data, Batch
from hashi_puzzle_solver.diffusion_utils import inject_noise
from hashi_puzzle_solver.diffusion_engine import DiffusionTrainer

@pytest.fixture
def sample_puzzle_data():
    # Simple 2-node puzzle
    x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float) # capacity, structural_degree, unused_capacity
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, 5), dtype=torch.float) # inv_dx, inv_dy, is_meta, bridge_label, is_labeled
    y = torch.tensor([1, 1], dtype=torch.long) # Ground truth: 1 bridge
    edge_mask = torch.tensor([True, True], dtype=torch.bool)
    node_type = torch.tensor([1, 1], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_mask=edge_mask, node_type=node_type)
    return data

def test_inject_noise_0(sample_puzzle_data):
    # 0 noise should mean input bridges == ground truth
    model_config = {
        "use_unused_capacity": True,
        "use_capacity": True,
        "use_structural_degree": True,
    }
    
    corrupted = inject_noise(
        sample_puzzle_data,
        noise_rate=0.0,
        bridge_label_idx=3,
        is_labeled_idx=4,
        model_config=model_config,
        device=torch.device("cpu")
    )
    
    assert torch.all(corrupted.edge_attr[:, 3] == sample_puzzle_data.y.float())
    assert torch.all(corrupted.edge_attr[:, 4] == 1.0)
    
    # unused_capacity should be original_capacity (2) - current_bridges_doubled (2*1) = 0
    assert torch.all(corrupted.x[:, 2] == 0.0)

def test_inject_noise_1(sample_puzzle_data):
    # 1.0 noise should flip all original edges
    model_config = {
        "use_unused_capacity": True,
        "use_capacity": True,
        "use_structural_degree": True,
    }
    
    corrupted = inject_noise(
        sample_puzzle_data,
        noise_rate=1.0,
        bridge_label_idx=3,
        is_labeled_idx=4,
        model_config=model_config,
        device=torch.device("cpu")
    )
    
    # Ground truth was 1, so corrupted should be 0 or 2
    assert torch.all(corrupted.edge_attr[:, 3] != sample_puzzle_data.y.float())
    assert torch.all(torch.isin(corrupted.edge_attr[:, 3], torch.tensor([0.0, 2.0])))
    assert torch.all(corrupted.edge_attr[:, 4] == 1.0)

def test_diffusion_trainer_step(sample_puzzle_data):
    # Mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3)
            self.use_verification_head = False
        def forward(self, x, edge_index, edge_attr=None, batch=None, node_type=None):
            # Return dummy logits [num_edges, 3]
            num_edges = edge_index.size(1)
            return torch.randn(num_edges, 3, requires_grad=True)

    config = {
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_edge_labels_as_features": True,
            "use_verification_head": False,
        },
        "training": {
            "loss_weights": None,
        }
    }
    
    device = torch.device("cpu")
    model = MockModel().to(device)
    trainer = DiffusionTrainer(model, config, device)
    
    batch = Batch.from_data_list([sample_puzzle_data])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    metrics = trainer.run_epoch(
        loader=[batch],
        epoch=1,
        total_epochs=1,
        optimizer=optimizer,
        training=True,
        noise_rate=0.5
    )
    
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] > 0
