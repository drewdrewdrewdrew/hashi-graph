"""Parity and execution test for the training loop."""

import torch
import pytest
from pathlib import Path
from hashi_puzzle_solver.engine import Trainer
from hashi_puzzle_solver.utils.common import load_config

_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


@pytest.mark.skip(reason="Temporarily disabled — slow / dataset-dependent parity step.")
def test_one_shot_training_step():
    """Verify that one training step can be executed without error."""
    config_path = _FIXTURES / "snapshot_diffusion_continuous.yaml"
    config = load_config(str(config_path))
    
    # Overrides for quick test
    config["data"]["limit"] = 2
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 2
    config["training"]["num_workers"] = 0 # Avoid multiprocessing issues in tests
    config["training"]["use_persistent_workers"] = False
    
    device = torch.device("cpu")
    # Force one-shot mode for this specific test
    config["training"]["mode"] = "one-shot"
    trainer = Trainer(config, device)
    
    # Check if dataset exists to avoid skipping
    raw_dir = Path(config["data"]["root_dir"]) / "raw"
    if not raw_dir.exists() or not list(raw_dir.glob("*.json")):
        pytest.skip("Dataset not found, skipping parity test.")

    # Run setup
    trainer._setup()
    
    # Run a single epoch (one-shot mode)
    trainer._trainer.current_masking_rate = 0.0
    metrics = trainer.run_epoch_one_shot(
        loader=trainer.train_loader,
        training=True,
    )
    
    assert metrics.loss >= 0
    assert metrics.accuracy >= 0
