import torch
import pytest
from hashi_puzzle_solver.models.transformer import TransformerEdgeClassifier
from hashi_puzzle_solver.diffusion_engine import DiffusionTrainer

def test_alpha_head_initialization():
    """Test model initialization with alpha head."""
    # Only alpha head
    model = TransformerEdgeClassifier(
        node_embedding_dim=16,
        hidden_channels=32,
        num_layers=2,
        use_meta_node=True,
        use_alpha_head=True,
        use_sigma_head=False
    )
    assert model.use_alpha_head is True
    assert model.use_sigma_head is False
    assert model.aux_out_channels == 1
    assert hasattr(model, "diffusion_aux_mlp")

    # Both heads
    model = TransformerEdgeClassifier(
        node_embedding_dim=16,
        hidden_channels=32,
        num_layers=2,
        use_meta_node=True,
        use_alpha_head=True,
        use_sigma_head=True
    )
    assert model.use_alpha_head is True
    assert model.use_sigma_head is True
    assert model.aux_out_channels == 2
    assert hasattr(model, "diffusion_aux_mlp")

def test_alpha_head_forward():
    """Test forward pass with alpha head."""
    model = TransformerEdgeClassifier(
        node_embedding_dim=16,
        hidden_channels=32,
        num_layers=2,
        use_meta_node=True,
        use_alpha_head=True,
        use_sigma_head=True
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

    # Test returning only alpha
    outputs = model(x, edge_index, node_type=node_type, return_alpha=True)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    logits, aux_logits = outputs
    assert logits.shape == (num_edges, 3)
    assert aux_logits.shape == (1, 2) # Both channels exist in aux_mlp

    # Test returning both
    outputs = model(x, edge_index, node_type=node_type, return_sigma=True, return_alpha=True)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    logits, aux_logits = outputs
    assert aux_logits.shape == (1, 2)

def test_trainer_alpha_loss():
    """Test trainer loss calculation for alpha head."""
    config = {
        "model": {
            "type": "transformer",
            "node_embedding_dim": 16,
            "hidden_channels": 32,
            "num_layers": 2,
            "use_meta_node": True,
            "use_alpha_head": True,
            "use_sigma_head": True,
            "use_unused_capacity": False,
        },
        "training": {
            "mode": "diff-cont",
            "loss_weights": {
                "ce": 1.0,
                "degree": 0.1,
                "crossing": 0.1,
                "verify": 0.0,
                "sigma": 0.1,
                "alpha": 0.1
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
    
    assert "alpha_loss" in metrics
    assert metrics["alpha_loss"] >= 0
    assert "sigma_loss" in metrics
    assert metrics["sigma_loss"] >= 0
