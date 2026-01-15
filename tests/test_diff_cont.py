import torch
import pytest
from torch_geometric.data import Data, Batch
from hashi_puzzle_solver.diffusion_utils import inject_continuous_noise
from hashi_puzzle_solver.diffusion_engine import DiffusionTrainer

@pytest.fixture
def sample_puzzle_data():
    # Simple 2-node puzzle
    # Node features: [capacity, structural_degree, unused_capacity]
    x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # Edge features: [inv_dx, inv_dy, is_meta, bridge_logits[3]]
    edge_attr = torch.zeros((2, 6), dtype=torch.float)
    y = torch.tensor([1, 1], dtype=torch.long) # Ground truth: 1 bridge
    edge_mask = torch.tensor([True, True], dtype=torch.bool)
    node_type = torch.tensor([1, 1], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_mask=edge_mask, node_type=node_type)
    return data

def test_inject_continuous_noise(sample_puzzle_data):
    model_config = {
        "use_unused_capacity": True,
        "use_capacity": True,
        "use_structural_degree": True,
    }
    
    # alpha=1.0 (full signal), sigma=0.0 (no noise), scale=10.0
    corrupted = inject_continuous_noise(
        sample_puzzle_data,
        alpha=1.0,
        sigma=0.0,
        scale=10.0,
        bridge_logits_idx=3,
        model_config=model_config,
        device=torch.device("cpu")
    )
    
    # bridge_logits_idx=3 starts a 3-wide block
    logits = corrupted.edge_attr[:, 3:6]
    
    # Ground truth y=1. Centered one-hot: [0, 1, 0] - 1/3 = [-1/3, 2/3, -1/3]
    # Scaled by 10: [-3.33, 6.66, -3.33]
    expected_signal = (torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]) - (1.0/3.0)) * 10.0
    assert torch.allclose(logits, expected_signal, atol=1e-5)
    
    # unused_capacity should be based on argmax of logits, which is label 1
    # capacity (2) - 1*1 = 1 (wait, update_node_features uses 2*label for unused capacity usually)
    # Let's check update_node_features behavior.
    # From previous tests, capacity 2 - bridges 1 -> unused 0 (because undirected?)
    # Actually hashi island capacity 'n' is usually compared to sum of bridges.
    # In my fixture, unused_capacity started at 2.0.
    
    # If argmax is 1, then unused should be 0.0 (assuming undirected edges each count 1)
    assert torch.all(corrupted.x[:, 2] == 0.0)

def test_diff_cont_trainer_step(sample_puzzle_data):
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1) # Add a parameter
            self.use_verification_head = False
            self.use_noise_head = True

        def forward(
            self,
            x,
            edge_index,
            edge_attr=None,
            batch=None,
            node_type=None,
            **kwargs
        ):
            num_edges = edge_index.size(1)
            num_graphs = (batch.max().item() + 1) if batch is not None else 1
            logits = torch.randn(num_edges, 3, requires_grad=True)
            noise_pred = torch.tensor([[0.5, 0.5]] * num_graphs, requires_grad=True)
            if kwargs.get("return_noise"):
                return logits, noise_pred
            return logits

    config = {
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_continuous_edge_labels": True,
            "use_verification_head": False,
            "use_noise_head": True,
        },
        "training": {
            "mode": "diff-cont",
            "sigma_max": 2.0,
            "scale_min": 4.0,
            "scale_max": 8.0,
            "loss_weights": {"ce": 1.0, "noise": 0.1},
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
        training=True
    )
    
    assert "loss" in metrics
    assert "noise_loss" in metrics
    assert metrics["noise_loss"] >= 0

def test_diff_cont_rollout(sample_puzzle_data):
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
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
            # Model always predicts label 1 (the ground truth)
            num_edges = edge_index.size(1)
            logits = torch.zeros(num_edges, 3)
            logits[:, 1] = 10.0 # High logit for label 1
            return logits

    config = {
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_continuous_edge_labels": True,
            "use_verification_head": False,
            "use_noise_head": False,
        },
        "training": {
            "mode": "diff-cont",
            "diffusion_step_lr": 1.0,
        }
    }
    
    device = torch.device("cpu")
    model = MockModel().to(device)
    trainer = DiffusionTrainer(model, config, device)
    
    batch = Batch.from_data_list([sample_puzzle_data])
    
    results = trainer.run_rollout(
        loader=[batch],
        max_steps=5,
        checkpoints=[1, 5]
    )
    
    assert "perfect_acc_k1" in results
    assert "perfect_acc_k5" in results
    assert results["perfect_acc_k1"] == 1.0 # Should be solved in 1 step with this mock

def test_diff_cont_rollout_flush(sample_puzzle_data):
    class PerfectModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_verification_head = False
            self.use_noise_head = False

        def forward(self, x, edge_index, **kwargs):
            num_edges = edge_index.size(1)
            logits = torch.zeros(num_edges, 3)
            logits[:, 1] = 10.0 # Right label
            return logits

    # Case 1: low LR, no flush -> not solved at step 1
    config_no_flush = {
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_continuous_edge_labels": True,
        },
        "training": {
            "mode": "diff-cont",
            "diffusion_step_lr": 0.01,
            "flush_first_step": False,
            "sigma_max": 2.0,
            "scale_max": 8.0,
        }
    }
    
    device = torch.device("cpu")
    torch.manual_seed(42) # Ensure deterministic noise for test
    model = PerfectModel().to(device)
    trainer = DiffusionTrainer(model, config_no_flush, device)
    batch = Batch.from_data_list([sample_puzzle_data])
    
    results = trainer.run_rollout(loader=[batch], max_steps=1, checkpoints=[1])
    # With sigma_max=2.0 and LR=0.01, the initial random noise will dominate.
    # acc = noise + 0.01 * (target - noise) = 0.99*noise + 0.01*target.
    # Probability of solving is very low.
    assert results["perfect_acc_k1"] < 1.0

    # Case 2: low LR, with flush -> solved at step 1
    config_flush = {
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_continuous_edge_labels": True,
        },
        "training": {
            "mode": "diff-cont",
            "diffusion_step_lr": 0.01,
            "flush_first_step": True,
            "sigma_max": 2.0,
            "scale_max": 8.0,
        }
    }
    trainer_flush = DiffusionTrainer(model, config_flush, device)
    results_flush = trainer_flush.run_rollout(loader=[batch], max_steps=1, checkpoints=[1])
    # With flush, step 1 LR is 1.0, so acc = target.
    assert results_flush["perfect_acc_k1"] == 1.0
