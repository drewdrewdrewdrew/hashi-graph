"""Tests for configuration loading and validation."""

from hashi_puzzle_solver.models.config import HashiModelConfig


def test_config_loading_from_yaml():
    """Test that we can load the diffusion_solver_continuous.yaml configuration."""
    config_path = "configs/diffusion_solver_continuous.yaml"
    config = HashiModelConfig.from_yaml(config_path)

    # Data Assertions
    assert config.data.limit == 1500
    assert config.data.root_dir == "dataset/"

    # Model Assertions
    assert config.model.type == "transformer"
    assert config.model.node_embedding_dim == 64
    assert config.model.hidden_channels == 128
    assert config.model.use_global_meta_node is True
    assert config.model.use_row_col_meta is True
    assert config.model.use_noise_head is True

    # Training Assertions
    assert config.training.mode == "diff-cont"
    assert config.training.learning_rate == 0.0001
    assert config.training.batch_size == 32
    assert config.training.loss_weights.ce == 1.67
    assert config.training.loss_weights.degree == 0.09
    assert config.training.early_stopping.monitor == "val_perfect_acc"


def test_config_to_dict_roundtrip():
    """Test that config to_dict and from_dict works correctly."""
    original_config = HashiModelConfig()
    config_dict = original_config.to_dict()
    new_config = HashiModelConfig.from_dict(config_dict)

    assert original_config == new_config
