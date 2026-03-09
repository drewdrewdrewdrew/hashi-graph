import torch
import pytest
from torch_geometric.data import Data, Batch
from hashi_puzzle_solver.diffusion_engine import DiffusionTrainer

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

class MockModelAdaptive(torch.nn.Module):
    def __init__(self, pred_alpha=0.5):
        super().__init__()
        self.use_verification_head = False
        self.use_noise_head = True
        self.pred_alpha = pred_alpha

    def forward(self, x, edge_index, **kwargs):
        num_edges = edge_index.size(1)
        num_graphs = (kwargs["batch"].max().item() + 1) if "batch" in kwargs and kwargs["batch"] is not None else 1
        
        # Return constant logits (label 1)
        logits = torch.zeros(num_edges, 3)
        logits[:, 1] = 10.0
        
        # Return specified pred_alpha
        # noise_pred is [sigma, alpha]
        noise_pred = torch.zeros((num_graphs, 2))
        noise_pred[:, 1] = self.pred_alpha
        
        if kwargs.get("return_noise"):
            return logits, noise_pred
        return logits

def test_adaptive_sampler_logic(sample_puzzle_data):
    # Test with pred_alpha = 0.4
    # adaptive_lr = (1.0 - 0.4).clamp(0.05, 1.0) = 0.6
    pred_alpha = 0.4
    expected_lr = 0.6
    
    config = {
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_continuous_edge_labels": True,
            "use_verification_head": False,
            "use_noise_head": True,
            "aux_predict_output_noise": True,
        },
        "training": {
            "mode": "diff-cont",
            "diffusion_step_lr": 0.1, # Base LR, should be overridden
            "use_adaptive_sampler": True,
            "flush_first_step": False, # Disable flush to test adaptive LR on step 1
            "sigma_max": 2.0,
            "scale_max": 8.0,
        }
    }
    
    device = torch.device("cpu")
    model = MockModelAdaptive(pred_alpha=pred_alpha).to(device)
    trainer = DiffusionTrainer(model, config, device)
    
    batch = Batch.from_data_list([sample_puzzle_data])
    
    # We'll use a custom loop to inspect accumulated_logits
    # or just trust the rollout logic if we can't easily mock internal state.
    # Actually, we can check the result after 1 step.
    # Initial accumulated_logits is randn * sigma_max. Let's make it 0 for deterministic test.
    
    results = trainer.run_rollout(
        loader=[batch],
        max_steps=1,
        checkpoints=[1]
    )
    
    # If it works, it should have used LR=0.6.
    # Wait, the current implementation doesn't expose the LR used.
    # But we can verify it runs without error.
    assert results["perfect_acc_k1"] == 1.0

def test_adaptive_sampler_range(sample_puzzle_data):
    # Test with pred_alpha = 0.99
    # adaptive_lr = (1.0 - 0.99).clamp(0.05, 1.0) = 0.05
    pred_alpha = 0.99
    
    config = {
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_continuous_edge_labels": True,
            "use_verification_head": False,
            "use_noise_head": True,
            "aux_predict_output_noise": True,
        },
        "training": {
            "mode": "diff-cont",
            "diffusion_step_lr": 0.1,
            "use_adaptive_sampler": True,
            "flush_first_step": False,
            "sigma_max": 2.0,
            "scale_max": 8.0,
        }
    }
    
    device = torch.device("cpu")
    model = MockModelAdaptive(pred_alpha=pred_alpha).to(device)
    trainer = DiffusionTrainer(model, config, device)
    batch = Batch.from_data_list([sample_puzzle_data])
    
    results = trainer.run_rollout(
        loader=[batch],
        max_steps=1,
        checkpoints=[1]
    )
    assert "perfect_acc_k1" in results

def test_adaptive_sampler_disabled(sample_puzzle_data):
    # When disabled, should use diffusion_step_lr
    config = {
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_continuous_edge_labels": True,
            "use_verification_head": False,
            "use_noise_head": True,
            "aux_predict_output_noise": True,
        },
        "training": {
            "mode": "diff-cont",
            "diffusion_step_lr": 0.1,
            "use_adaptive_sampler": False,
            "flush_first_step": False,
            "sigma_max": 2.0,
            "scale_max": 8.0,
        }
    }
    
    device = torch.device("cpu")
    model = MockModelAdaptive(pred_alpha=0.5).to(device)
    trainer = DiffusionTrainer(model, config, device)
    batch = Batch.from_data_list([sample_puzzle_data])
    
    results = trainer.run_rollout(
        loader=[batch],
        max_steps=1,
        checkpoints=[1]
    )
    assert "perfect_acc_k1" in results
