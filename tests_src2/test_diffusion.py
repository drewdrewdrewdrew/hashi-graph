import torch
import pytest
from torch_geometric.data import Data, Batch
from hashi_puzzle_solver.utils.diffusion_utils import inject_noise
from hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer

@pytest.fixture
def sample_puzzle_data():
    # Simple 2-node puzzle
    # capacity, structural_degree, unused_capacity
    x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float) 
    # Bidirectional: 0-1 and 1-0
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # inv_dx, inv_dy, is_meta, bridge_label, is_labeled
    edge_attr = torch.zeros((2, 5), dtype=torch.float) 
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
    
    # unused_capacity should be original_capacity (2) - current_bridges = 2 - 1 = 1
    # Wait, in Hashi a bridge count of 1 means 1 bridge. 
    # unused_capacity = capacity - degree.
    # Island 0 has one neighbor 1 with 1 bridge. So degree is 1.
    # unused_capacity = 2 - 1 = 1.
    assert torch.all(corrupted.x[:, 2] == 1.0)

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
            self.lin = torch.nn.Linear(1, 1) # dummy param
            self.use_verification_head = False
            self.use_noise_head = False

        def forward(
            self,
            x,
            edge_index,
            edge_attr=None,
            batch=None,
            node_type=None,
            **kwargs
        ):
            # Return dummy logits [num_edges, 3]
            num_edges = edge_index.size(1)
            logits = torch.randn(num_edges, 3, device=x.device, requires_grad=True)
            
            res = [logits]
            if kwargs.get("return_verification"):
                res.append(torch.randn(x.size(0) if batch is None else batch.max() + 1, 1, device=x.device, requires_grad=True))
            if kwargs.get("return_noise"):
                res.append(torch.randn(x.size(0) if batch is None else batch.max() + 1, 2, device=x.device, requires_grad=True))
            
            if len(res) == 1:
                return res[0]
            return tuple(res)

    config = {
        "data": {
            "root_dir": "dataset/",
        },
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_edge_labels_as_features": True,
            "use_verification_head": False,
            "use_noise_head": False,
        },
        "training": {
            "mode": "diff-discrete",
            "loss_weights": {"ce": 1.0, "degree": 0.0, "crossing": 0.0, "verify": 0.0},
            "num_inference_steps_training": 1,
            "learning_rate": 0.001,
            "batch_size": 1,
        }
    }
    
    device = torch.device("cpu")
    
    # BaseTrainer requires config, device
    trainer = DiffusionTrainer(config, device)
    trainer.model = MockModel().to(device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    
    batch = Batch.from_data_list([sample_puzzle_data])
    
    metrics = trainer.run_epoch(
        loader=[batch],
        epoch=1,
        total_epochs=1,
        training=True,
        noise_rate=0.5
    )
    
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] != 0
