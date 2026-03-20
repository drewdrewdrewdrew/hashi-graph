import torch
import pytest
from torch_geometric.data import Data, Batch
from hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer
from hashi_puzzle_solver.models.transformer import TransformerEdgeClassifier

@pytest.fixture
def config():
    return {
        "model": {
            "type": "transformer",
            "node_embedding_dim": 16,
            "hidden_channels": 32,
            "num_layers": 2,
            "heads": 2,
            "dropout": 0.0,
            "use_global_meta_node": True,
            "use_row_col_meta": False,
            "use_meta_mesh": False,
            "use_meta_row_col_edges": False,
            "use_component_meta": False,
            "edge_concat_global_meta": True,
            "use_edge_features_in_prediction": True,
            "use_distance": True,
            "use_conflict_edges": False,
            "use_potential_crossing": False,
            "use_edge_labels_as_features": False,
            "use_continuous_edge_labels": True,
            "use_cut_edges": False,
            "use_structural_degree": True,
            "use_capacity": True,
            "use_unused_capacity": True,
            "use_conflict_status": False,
            "use_closeness_centrality": False,
            "use_articulation_points": False,
            "use_spectral_features": False,
            "use_verification_head": False,
            "use_noise_head": True,
            "aux_predict_output_noise": False,
        },
        "training": {
            "mode": "diff-cont",
            "learning_rate": 0.001,
            "num_inference_steps_training": 2,
            "sigma_max": 2.0,
            "scale_min": 4.0,
            "scale_max": 8.0,
            "alpha_power": 1.0,
            "zero_signal_prob": 0.1,
            "diffusion_step_lr": 0.5,
            "loss_weights": {
                "ce": 1.0,
                "degree": 0.1,
                "crossing": 0.1,
                "noise": 0.1,
            }
        }
    }

@pytest.fixture
def sample_data():
    # Simple 2-node puzzle with global meta node (node_type 9)
    # 2 nodes, 1 puzzle edge, 2 meta edges (global meta to nodes)
    x = torch.zeros((3, 10), dtype=torch.float)
    x[0, 0] = 1 # capacity node 1
    x[1, 0] = 1 # capacity node 2
    x[2, 0] = 9 # capacity global meta (convention)
    
    node_type = torch.tensor([1, 1, 9], dtype=torch.long)
    
    # edge_index: [0, 1] is puzzle edge, [2, 0], [2, 1] are meta edges
    edge_index = torch.tensor([[0, 1, 1, 0, 2, 0, 2, 1, 0, 2, 1, 2], 
                               [1, 0, 0, 1, 0, 2, 1, 2, 2, 0, 2, 1]], dtype=torch.long)
    
    # Edge attr: base dim is 3: [inv_dx, inv_dy, is_meta]
    # use_continuous_edge_labels adds 3 -> total 6
    edge_attr = torch.zeros((edge_index.size(1), 6), dtype=torch.float)
    edge_attr[:4, 2] = 0 # puzzle edges
    edge_attr[4:, 2] = 1 # meta edges
    
    y = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    edge_mask = torch.tensor([True, True, True, True, False, False, False, False, False, False, False, False], dtype=torch.bool)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_mask=edge_mask, node_type=node_type)
    data.num_graphs = 1
    return data

def test_multi_step_training_and_noise_head(config, sample_data):
    device = torch.device("cpu")
    model = TransformerEdgeClassifier(
        node_embedding_dim=config["model"]["node_embedding_dim"],
        hidden_channels=config["model"]["hidden_channels"],
        num_layers=config["model"]["num_layers"],
        heads=config["model"]["heads"],
        use_meta_node=True,
        edge_dim=6,
        use_continuous_edge_labels=True,
        use_noise_head=True,
        edge_concat_global_meta=True,
        use_edge_features_in_prediction=True
    ).to(device)
    
    trainer = DiffusionTrainer(model, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch = Batch.from_data_list([sample_data])
    
    # Run one epoch
    metrics = trainer.run_epoch(
        loader=[batch],
        epoch=1,
        total_epochs=1,
        optimizer=optimizer,
        training=True
    )
    
    assert "loss" in metrics
    assert "noise_loss" in metrics
    assert metrics["loss"] > 0
    assert metrics["noise_loss"] >= 0

def test_aux_predict_output_noise(config, sample_data):
    # Set aux_predict_output_noise to True
    config["model"]["aux_predict_output_noise"] = True
    
    device = torch.device("cpu")
    model = TransformerEdgeClassifier(
        node_embedding_dim=config["model"]["node_embedding_dim"],
        hidden_channels=config["model"]["hidden_channels"],
        num_layers=config["model"]["num_layers"],
        heads=config["model"]["heads"],
        use_meta_node=True,
        edge_dim=6,
        use_continuous_edge_labels=True,
        use_noise_head=True,
        edge_concat_global_meta=True,
        use_edge_features_in_prediction=True
    ).to(device)
    
    trainer = DiffusionTrainer(model, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch = Batch.from_data_list([sample_data])
    
    # Run one epoch
    metrics = trainer.run_epoch(
        loader=[batch],
        epoch=1,
        total_epochs=1,
        optimizer=optimizer,
        training=True
    )
    
    assert "noise_loss" in metrics
    assert metrics["noise_loss"] >= 0

def test_rollout_noise_feedback(config, sample_data):
    # Set use_noise_head and aux_predict_output_noise to True
    config["model"]["use_noise_head"] = True
    config["model"]["aux_predict_output_noise"] = True
    
    device = torch.device("cpu")
    model = TransformerEdgeClassifier(
        node_embedding_dim=config["model"]["node_embedding_dim"],
        hidden_channels=config["model"]["hidden_channels"],
        num_layers=config["model"]["num_layers"],
        heads=config["model"]["heads"],
        use_meta_node=True,
        edge_dim=6,
        use_continuous_edge_labels=True,
        use_noise_head=True,
        edge_concat_global_meta=True,
        use_edge_features_in_prediction=True
    ).to(device)
    
    trainer = DiffusionTrainer(model, config, device)
    batch = Batch.from_data_list([sample_data])
    
    # Run rollout
    # We want to verify that it runs without error and that current_input_noise
    # is being updated (though we can't easily check internal state without mocking)
    # But we can at least ensure the logic path is covered.
    results = trainer.run_rollout(
        loader=[batch],
        max_steps=2,
        checkpoints=[1, 2]
    )
    
    assert "perfect_acc_k1" in results
    assert "perfect_acc_k2" in results
    assert "accuracy" in results
