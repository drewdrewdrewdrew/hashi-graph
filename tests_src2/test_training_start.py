"""End-to-end smoke test for training start."""

import torch
from hashi_puzzle_solver.engine import Trainer


def test_trainer_setup():
    """Test that the trainer can be initialized and set up with the new model architecture."""
    config = {
        "data": {
            "root_dir": "dataset/",
            "limit": 10
        },
        "model": {
            "type": "transformer",
            "node_embedding_dim": 16,
            "hidden_channels": 32,
            "num_layers": 2,
            "heads": 4,
            "dropout": 0.1,
            "use_global_meta_node": True,
            "use_row_col_meta": False,
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_conflict_status": True,
            "use_categorical_edge_types": True
        },
        "training": {
            "mode": "one-shot",
            "learning_rate": 0.001,
            "batch_size": 2,
            "epochs": 1,
            "loss_weights": {
                "ce": 1.0,
                "degree": 0.1,
                "crossing": 0.1
            }
        }
    }
    
    device = torch.device("cpu")
    trainer = Trainer(config, device)
    
    # This should call ModelFactory.create_model and set up optimizer
    # We might need to mock dataloaders if we don't want to rely on actual data files.
    # But for a smoke test, initialization is a good start.
    
    trainer._setup()
    
    assert trainer.model is not None
    assert trainer.optimizer is not None
    # Check if the model is our new HashiGraphModel
    from hashi_puzzle_solver.models.core import HashiGraphModel
    assert isinstance(trainer.model, HashiGraphModel)
