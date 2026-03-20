import torch
import pytest
from torch_geometric.data import Data, Batch
from hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer

@pytest.fixture
def sample_puzzle_data():
    # Simple 2-node puzzle
    x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, 6), dtype=torch.float)
    y = torch.tensor([1, 1], dtype=torch.long)
    edge_mask = torch.tensor([True, True], dtype=torch.bool)
    node_type = torch.tensor([1, 1], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_mask=edge_mask, node_type=node_type)
    return data

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)
        self.use_verification_head = False
        self.use_noise_head = False

    def forward(self, x, edge_index, **kwargs):
        num_edges = edge_index.size(1)
        # Return something that requires grad
        dummy_input = torch.ones((num_edges, 3), device=x.device)
        logits = self.lin(dummy_input)
        return logits

def test_recursive_buffer_initialization():
    config = {
        "model": {"bridge_logits": 3, "bridge_label": 0, "is_labeled": 1},
        "training": {"mode": "diff-cont", "recursive_carryover": True}
    }
    trainer = DiffusionTrainer(MockModel(), config, torch.device("cpu"))
    assert hasattr(trainer, "carry_over_buffer_train")
    assert hasattr(trainer, "carry_over_buffer_val")
    assert isinstance(trainer.carry_over_buffer_train, list)
    assert len(trainer.carry_over_buffer_train) == 0

def test_buffer_refill(sample_puzzle_data):
    config = {
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_continuous_edge_labels": True,
            "use_verification_head": False,
            "use_noise_head": False,
            "bridge_logits": 3,
        },
        "training": {
            "mode": "diff-cont",
            "recursive_carryover": True,
            "sigma_max": 2.0,
            "scale_min": 4.0,
            "scale_max": 8.0,
            "zero_signal_prob": 0.0, # Target all puzzles for carry-over
            "loss_weights": {"ce": 1.0},
        }
    }
    
    device = torch.device("cpu")
    model = MockModel().to(device)
    trainer = DiffusionTrainer(model, config, device)
    
    batch = Batch.from_data_list([sample_puzzle_data, sample_puzzle_data])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Run one batch (training)
    trainer.run_epoch(
        loader=[batch],
        epoch=1,
        total_epochs=1,
        optimizer=optimizer,
        training=True
    )
    
    assert len(trainer.carry_over_buffer_train) == 2
    assert len(trainer.carry_over_buffer_val) == 0
    
    # Run one batch (validation)
    trainer.run_epoch(
        loader=[batch],
        epoch=1,
        total_epochs=1,
        optimizer=optimizer,
        training=False
    )
    
    assert len(trainer.carry_over_buffer_val) == 2

def test_batch_mixing(sample_puzzle_data):
    config = {
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_continuous_edge_labels": True,
            "use_verification_head": False,
            "use_noise_head": False,
            "bridge_logits": 3,
        },
        "training": {
            "mode": "diff-cont",
            "recursive_carryover": True,
            "sigma_max": 2.0,
            "scale_min": 4.0,
            "scale_max": 8.0,
            "zero_signal_prob": 0.5, # 50% fresh, 50% carry
            "loss_weights": {"ce": 1.0},
        }
    }
    
    device = torch.device("cpu")
    model = MockModel().to(device)
    trainer = DiffusionTrainer(model, config, device)
    
    # Manually seed train buffer with 2 puzzles
    p1 = sample_puzzle_data.clone()
    p1.edge_attr[:, 3:6] = 1.0 # Distinctive value
    p2 = sample_puzzle_data.clone()
    p2.edge_attr[:, 3:6] = 2.0
    
    # New buffer format: (Data, noise_stats, scale)
    # [sigma, alpha]. Set alpha=0.5 to distinguish from fresh puzzles (alpha=0)
    noise_stats = torch.tensor([0.5, 0.5], device=device)
    scale = torch.tensor(8.0, device=device)
    trainer.carry_over_buffer_train = [(p1, noise_stats, scale), (p2, noise_stats, scale)]
    
    # Incoming batch of 4
    batch = Batch.from_data_list([sample_puzzle_data] * 4)
    
    # Mocking _prepare_mixed_batch to check its output for training
    mixed_batch, alphas, sigmas, scales, n = trainer._prepare_mixed_batch(batch, config["training"], training=True)
    
    assert n == 4
    assert mixed_batch.num_graphs == 4
    assert torch.all(alphas[:2] == 0.0) # Fresh
    assert not torch.all(alphas[2:] == 0.0) # Carry-over

def test_buffer_limit(sample_puzzle_data):
    config = {
        "model": {"bridge_logits": 3},
        "training": {"mode": "diff-cont", "zero_signal_prob": 0.0, "recursive_carryover": True}
    }
    trainer = DiffusionTrainer(MockModel(), config, torch.device("cpu"))
    
    # Fill buffer beyond limit (limit is 4x batch_size = 40)
    batch = Batch.from_data_list([sample_puzzle_data] * 10)
    trainer._refill_buffer(batch, torch.randn(20, 3), torch.ones(10) * 8.0, config["training"], training=True)
    trainer._refill_buffer(batch, torch.randn(20, 3), torch.ones(10) * 8.0, config["training"], training=True)
    trainer._refill_buffer(batch, torch.randn(20, 3), torch.ones(10) * 8.0, config["training"], training=True)
    trainer._refill_buffer(batch, torch.randn(20, 3), torch.ones(10) * 8.0, config["training"], training=True)
    trainer._refill_buffer(batch, torch.randn(20, 3), torch.ones(10) * 8.0, config["training"], training=True)
    
    assert len(trainer.carry_over_buffer_train) <= 40
