"""Tests for configuration loading and validation."""

import pathlib

from hashi_puzzle_solver.models.config import HashiModelConfig

_FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def test_config_loading_from_yaml():
    """Load a frozen minimal YAML fixture (not the live training config)."""
    config_path = _FIXTURES / "minimal_hashi_config.yaml"
    config = HashiModelConfig.from_yaml(config_path)

    assert config.data.limit == 42
    assert config.data.root_dir == "mock_dataset/"
    assert config.model.type == "transformer"
    assert config.model.node_embedding_dim == 16
    assert config.model.hidden_channels == 32
    assert config.model.use_noise_head is False
    assert config.training.mode == "diff-cont"
    assert config.training.learning_rate == 0.0001
    assert config.training.batch_size == 7
    assert config.training.loss_weights.ce == 1.0
    assert config.training.loss_weights.degree == 0.1
    assert config.training.early_stopping.monitor == "val_loss"


def test_config_to_dict_roundtrip():
    """Test that config to_dict and from_dict works correctly."""
    original_config = HashiModelConfig()
    config_dict = original_config.to_dict()
    new_config = HashiModelConfig.from_dict(config_dict)

    assert original_config == new_config
