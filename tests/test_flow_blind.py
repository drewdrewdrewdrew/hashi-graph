import torch
import pytest
from pathlib import Path
from hashi_puzzle_solver.utils.common import load_config
from hashi_puzzle_solver.models.config import HashiModelConfig
from hashi_puzzle_solver.models.factory import ModelFactory
from hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer
from hashi_puzzle_solver.engine import Trainer

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_flow_blind_initialization():
    """Test that the flow-blind mode can be initialized and runs a forward pass."""
    config_path = _FIXTURES / "snapshot_flow_blind.yaml"
    if not config_path.is_file():
        pytest.skip("Flow-blind fixture not found")
    
    config = load_config(str(config_path))
    device = torch.device("cpu")
    hcfg = HashiModelConfig.from_dict(config)

    # Check model creation
    model = ModelFactory.create_model(hcfg, device)
    assert hcfg.model.use_time_conditioning is True
    assert model is not None

    # Check trainer creation
    trainer = Trainer(config, device)
    assert trainer is not None

def test_flow_blind_training_step():
    """Test a single training step in flow-blind mode."""
    config_path = _FIXTURES / "snapshot_flow_blind.yaml"
    if not config_path.is_file():
        pytest.skip("Flow-blind fixture not found")
    
    config = load_config(str(config_path))
    config["data"]["limit"] = 10
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 2

    device = torch.device("cpu")
    trainer = Trainer(config, device)

    # We don't actually need to run the full trainer.train()
    # as it requires a dataset. But we can test the DiffusionTrainer.
    hcfg = HashiModelConfig.from_dict(config)
    model = ModelFactory.create_model(hcfg, device)
    diff_trainer = DiffusionTrainer(config, device)
    diff_trainer.model = model
    
    # Mock a batch
    from torch_geometric.data import Data, Batch
    # 2 islands + 1 global meta node = 3 nodes
    edge_index = torch.tensor([[0, 1, 1, 0, 0, 2, 1, 2], [1, 0, 0, 1, 2, 0, 2, 1]], dtype=torch.long)
    x = torch.zeros((3, 11))
    x[0:2, 0] = torch.tensor([1, 1]) # capacities
    x[2, 0] = 9 # global meta node type
    y = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.long)
    edge_mask = torch.tensor([True, True, True, True, False, False, False, False])
    node_type = torch.tensor([0, 0, 9])
    
    data = Data(x=x, edge_index=edge_index, y=y, edge_mask=edge_mask, node_type=node_type)
    data.num_nodes = 3
    edim = diff_trainer.model.edge_encoder.output_dim
    data.edge_attr = torch.zeros((8, edim))
    
    batch = Batch.from_data_list([data, data])
    
    # Run one epoch (training=True)
    loader = [batch]
    diff_trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    metrics = diff_trainer.run_epoch(loader, epoch=1, total_epochs=1, training=True)
    assert "loss" in metrics
    assert metrics["loss"] > 0

def test_flow_blind_rollout():
    """Test the rollout mechanism for flow-blind mode."""
    config_path = _FIXTURES / "snapshot_flow_blind.yaml"
    if not config_path.is_file():
        pytest.skip("Flow-blind fixture not found")
    
    config = load_config(str(config_path))
    device = torch.device("cpu")
    hcfg = HashiModelConfig.from_dict(config)
    model = ModelFactory.create_model(hcfg, device)
    diff_trainer = DiffusionTrainer(config, device)
    diff_trainer.model = model
    
    # Mock a batch
    from torch_geometric.data import Data, Batch
    # 2 islands + 1 global meta node = 3 nodes
    edge_index = torch.tensor([[0, 1, 1, 0, 0, 2, 1, 2], [1, 0, 0, 1, 2, 0, 2, 1]], dtype=torch.long)
    x = torch.zeros((3, 11))
    x[0:2, 0] = torch.tensor([1, 1])
    x[2, 0] = 9
    y = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.long)
    edge_mask = torch.tensor([True, True, True, True, False, False, False, False])
    node_type = torch.tensor([0, 0, 9])
    
    data = Data(x=x, edge_index=edge_index, y=y, edge_mask=edge_mask, node_type=node_type)
    data.num_nodes = 3
    edim = model.edge_encoder.output_dim
    data.edge_attr = torch.zeros((8, edim))
    
    batch = Batch.from_data_list([data])
    loader = [batch]
    
    # Run rollout
    rollout_results = diff_trainer.run_rollout(loader, max_steps=5, checkpoints=[1, 5])
    assert "perfect_acc_k5" in rollout_results
    assert "accuracy" in rollout_results
