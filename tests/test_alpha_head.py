import torch
import pytest
from hashi_puzzle_solver.models.transformer import TransformerEdgeClassifier
from hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer

def test_noise_head_initialization():
    """Test model initialization with noise head."""
    # With noise head
    model = TransformerEdgeClassifier(
        node_embedding_dim=16,
        hidden_channels=32,
        num_layers=2,
        use_meta_node=True,
        use_noise_head=True
    )
    assert model.use_noise_head is True
    assert hasattr(model, "diffusion_aux_mlp")
    assert model.diffusion_aux_mlp[-1].out_features == 2 # (sigma, alpha)

def test_noise_head_forward():
    """Test forward pass with noise head."""
    model = TransformerEdgeClassifier(
        node_embedding_dim=16,
        hidden_channels=32,
        num_layers=2,
        use_meta_node=True,
        use_noise_head=True
    )
    model.eval()

    num_nodes = 10
    num_edges = 20
    # Create x with valid indices for categorical features
    x = torch.zeros((num_nodes, 16))
    # Column 0: Capacity (0-15)
    x[:, 0] = torch.randint(0, 16, (num_nodes,)).float()
    # Column 1: Degree (0-15)
    x[:, 1] = torch.randint(0, 16, (num_nodes,)).float()
    # Ensure at least one global meta node (type 9)
    x[0, 0] = 9 
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    node_type = torch.zeros(num_nodes, dtype=torch.long)
    node_type[0] = 9
    
    # We need a batch tensor for global_mean_pool
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Test returning noise
    outputs = model(x, edge_index, batch=batch, node_type=node_type, return_noise=True)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    logits, noise_logits = outputs
    assert logits.shape == (num_edges, 3)
    assert noise_logits.shape == (1, 2) # (sigma, alpha)

def test_trainer_noise_loss():
    """Test trainer loss calculation for noise head."""
    config = {
        "model": {
            "type": "transformer",
            "node_embedding_dim": 16,
            "hidden_channels": 32,
            "num_layers": 2,
            "use_meta_node": True,
            "use_noise_head": True,
            "use_unused_capacity": False,
        },
        "training": {
            "mode": "diff-cont",
            "loss_weights": {
                "ce": 1.0,
                "degree": 0.1,
                "crossing": 0.1,
                "verify": 0.0,
                "noise": 0.1,
            },
            "alpha_power": 1.0,
            "zero_signal_prob": 0.0,
            "sigma_max": 2.0,
            "scale_min": 4.0,
            "scale_max": 8.0
        }
    }
    
    device = torch.device("cpu")
    model = TransformerEdgeClassifier(**config["model"]).to(device)
    trainer = DiffusionTrainer(model, config, device)
    
    # Mock data batch
    from torch_geometric.data import Data, Batch
    num_nodes = 5
    num_edges = 8
    x = torch.zeros((num_nodes, 16))
    x[:, 0] = torch.randint(0, 16, (num_nodes,)).float()
    x[:, 1] = torch.randint(0, 16, (num_nodes,)).float()
    
    data = Data(
        x=x,
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.zeros((num_edges, 3)), # Match edge_dim=3
        node_type=torch.tensor([9, 1, 1, 1, 1]),
        y=torch.randint(0, 3, (num_edges,)),
        edge_mask=torch.ones(num_edges, dtype=torch.bool),
        num_nodes=num_nodes,
        batch=torch.zeros(num_nodes, dtype=torch.long)
    )
    # Mock some indices for trainer
    trainer.bridge_logits_idx = 0
    
    loader = [data]
    
    # Run one step
    metrics = trainer.run_epoch(loader, 0, 1, training=True)
    
    assert "noise_loss" in metrics
    assert metrics["noise_loss"] >= 0
