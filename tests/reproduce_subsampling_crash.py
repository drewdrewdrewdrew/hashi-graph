import torch
from torch_geometric.data import Data, Batch
from hashi_puzzle_solver.utils.common import custom_collate_with_conflicts
from hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer

def test_diffusion_subsampling_multistep():
    # Mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)
            self.use_verification_head = False
            self.use_noise_head = False
            
        def forward(self, x, edge_index, **kwargs):
            return torch.randn(edge_index.size(1), 3, requires_grad=True)

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
            "num_inference_steps_training": 2, # Multi-step
            "n_blocks": 1, # Triggers subsampling
            "loss_weights": {"ce": 1.0, "crossing": 1.0, "degree": 1.0},
        }
    }
    
    device = torch.device("cpu")
    model = MockModel().to(device)
    trainer = DiffusionTrainer(model, config, device)
    
    # Create a batch of 4 graphs
    data_list = []
    for _ in range(4):
        # Node features: [capacity, structural_degree, unused_capacity]
        x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        # Edge features: [inv_dx, inv_dy, is_meta, bridge_label, is_labeled, bridge_logits[3]]
        edge_attr = torch.zeros((2, 9), dtype=torch.float)
        y = torch.tensor([1, 1])
        edge_mask = torch.tensor([True, True])
        node_type = torch.tensor([1, 1, 1])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_mask=edge_mask, node_type=node_type)
        data_list.append(data)
    
    batch = custom_collate_with_conflicts(data_list)
    
    # Mock optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # This should not crash
    metrics = trainer.run_epoch(
        loader=[batch],
        epoch=1,
        total_epochs=1,
        optimizer=optimizer,
        training=True
    )
    
    print("Metrics:", metrics)
    assert "loss" in metrics

if __name__ == "__main__":
    test_diffusion_subsampling_multistep()
