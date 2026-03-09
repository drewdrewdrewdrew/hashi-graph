import torch
import pytest
from hashi_puzzle_solver.models.factory import ModelFactory
from hashi_puzzle_solver.models.transformer import TransformerEdgeClassifier
from hashi_puzzle_solver.diffusion_engine import DiffusionTrainer
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

def test_mlp_configurability():
    """Test that MLP width and depth multipliers work as expected."""
    base_config = {
        "model": {
            "type": "transformer",
            "node_embedding_dim": 64,
            "hidden_channels": 128,
            "num_layers": 2,
            "heads": 4,
            "dropout": 0.1,
            "use_global_meta_node": True,
            "edge_mlp_width_mult": 2,
            "edge_mlp_depth_mult": 2,
        }
    }
    
    device = torch.device("cpu")
    model = ModelFactory.create_model(base_config, device)
    
    assert isinstance(model, TransformerEdgeClassifier)
    
    # Check MLP structure
    # Expected: Sequential(
    #   Linear(input_dim, 256), ReLU, Dropout,
    #   Linear(256, 256), ReLU, Dropout,
    #   Linear(256, 3)
    # )
    mlp = model.edge_mlp
    assert len(mlp) == 3 * 2 + 1  # 2 * (Linear, ReLU, Dropout) + 1 Linear
    
    # Check hidden layer dimensions
    assert mlp[0].out_features == 256
    assert mlp[3].in_features == 256
    assert mlp[3].out_features == 256
    assert mlp[6].in_features == 256
    assert mlp[6].out_features == 3

def test_node_noise_mlp_configurability():
    """Test that Node Encoder and Noise Head MLP multipliers work as expected."""
    config = {
        "model": {
            "type": "transformer",
            "node_embedding_dim": 64,
            "hidden_channels": 128,
            "num_layers": 2,
            "heads": 4,
            "dropout": 0.1,
            "use_global_meta_node": True,
            "use_noise_head": True,
            "node_encoder_width_mult": 1.5,
            "node_encoder_depth_mult": 2,
            "noise_mlp_width_mult": 2.0,
            "noise_mlp_depth_mult": 3,
        }
    }
    
    device = torch.device("cpu")
    model = ModelFactory.create_model(config, device)
    
    # Check Node Encoder MLP
    # NodeEncoder refinement MLP structure: 
    # [Linear(total_input_dim, 192), LayerNorm, ReLU] * 2 + Linear(192, 128)
    # Total layers: 2 * 3 + 1 = 7
    node_refiner = model.node_encoder.refiner
    assert len(node_refiner) == 7
    assert node_refiner[0].out_features == 192  # 128 * 1.5
    assert node_refiner[3].out_features == 192
    assert node_refiner[6].out_features == 128
    
    # Check Noise Head MLP
    # Noise MLP structure:
    # [Linear(input_dim, 256), ReLU, Dropout] * 3 + Linear(256, 2)
    # Total layers: 3 * 3 + 1 = 10
    noise_mlp = model.diffusion_aux_mlp
    assert len(noise_mlp) == 10
    assert noise_mlp[0].out_features == 256  # 128 * 2.0
    assert noise_mlp[3].out_features == 256
    assert noise_mlp[6].out_features == 256
    assert noise_mlp[9].out_features == 2

def test_dynamic_training_loop_steps():
    """Test that the multi-step training loop with dynamic noise alignment runs."""
    config = {
        "model": {
            "type": "transformer",
            "node_embedding_dim": 16,
            "hidden_channels": 32,
            "num_layers": 2,
            "heads": 2,
            "use_global_meta_node": True,
            "use_noise_head": True,
            "use_continuous_edge_labels": True,
        },
        "training": {
            "mode": "diff-cont",
            "num_inference_steps_training": 2,
            "loss_weights": {"ce": 1.0, "degree": 0.1, "crossing": 0.1, "noise": 0.1},
            "sigma_max": 2.0,
            "alpha_power": 1.0,
            "zero_signal_prob": 0.1,
            "scale_min": 4.0,
            "scale_max": 8.0,
        }
    }
    
    device = torch.device("cpu")
    model = ModelFactory.create_model(config, device)
    trainer = DiffusionTrainer(model, config, device)
    
    # Mock data
    num_nodes = 5
    num_edges = 10
    
    # Node features: 4 columns [capacity, degree, unused, conflict]
    x = torch.zeros((num_nodes, 4))
    x[:, 0] = torch.randint(1, 9, (num_nodes,))  # capacity
    x[:, 1] = torch.randint(1, 5, (num_nodes,))  # degree
    x[:, 2] = torch.randn(num_nodes)              # unused
    x[:, 3] = torch.randint(0, 2, (num_nodes,))  # conflict
    
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.zeros(num_edges, model.edge_dim)
    y = torch.randint(0, 3, (num_edges,))
    edge_mask = torch.ones(num_edges, dtype=torch.bool)
    
    # Global meta node is type 9
    node_type = torch.zeros(num_nodes).long()
    node_type[0] = 9 
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_mask=edge_mask, node_type=node_type)
    batch = Batch.from_data_list([data])
    
    loader = [batch]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # This should run without error
    metrics = trainer.run_epoch(loader, epoch=1, total_epochs=1, optimizer=optimizer, training=True)
    
    assert "loss" in metrics
    assert "noise_loss" in metrics
    assert metrics["loss"] > 0
