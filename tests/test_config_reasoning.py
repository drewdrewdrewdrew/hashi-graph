"""Tests for ReasoningConfig, ReverseGnnConfig, and their integration into ModelConfig and HashiModelConfig."""

import pytest

from src2.hashi_puzzle_solver.models.config import (
    HashiModelConfig,
    ModelConfig,
    ReasoningConfig,
    ReverseGnnConfig,
)


# ---------------------------------------------------------------------------
# ReasoningConfig tests
# ---------------------------------------------------------------------------

def test_reasoning_config_defaults():
    """Test 1: ReasoningConfig() has enabled=False, steps=5."""
    cfg = ReasoningConfig()
    assert cfg.enabled is False
    assert cfg.steps == 5


def test_reasoning_config_steps_zero_raises():
    """Test 2: ReasoningConfig(steps=0) raises ValueError."""
    with pytest.raises(ValueError, match="reasoning.steps must be >= 1"):
        ReasoningConfig(steps=0)


def test_reasoning_config_steps_one_ok():
    """Test 3: ReasoningConfig(steps=1) does not raise."""
    cfg = ReasoningConfig(steps=1)
    assert cfg.steps == 1


# ---------------------------------------------------------------------------
# ReverseGnnConfig tests
# ---------------------------------------------------------------------------

def test_reverse_gnn_config_defaults():
    """Test 4: ReverseGnnConfig() has enabled=False, separate_weights=True, project_embeddings=True."""
    cfg = ReverseGnnConfig()
    assert cfg.enabled is False
    assert cfg.separate_weights is True
    assert cfg.project_embeddings is True


def test_reverse_gnn_config_no_post_init():
    """Test 5: ReverseGnnConfig has no __post_init__ (all booleans, no validation)."""
    # Should be able to construct with arbitrary bool values without raising
    cfg = ReverseGnnConfig(enabled=True, separate_weights=False, project_embeddings=False)
    assert cfg.enabled is True
    assert cfg.separate_weights is False
    assert cfg.project_embeddings is False


# ---------------------------------------------------------------------------
# ModelConfig integration tests
# ---------------------------------------------------------------------------

def test_model_config_has_reasoning_field():
    """Test 6: ModelConfig() has reasoning attribute of type ReasoningConfig with enabled=False."""
    cfg = ModelConfig()
    assert hasattr(cfg, "reasoning")
    assert isinstance(cfg.reasoning, ReasoningConfig)
    assert cfg.reasoning.enabled is False


def test_model_config_has_reverse_gnn_field():
    """Test 7: ModelConfig() has reverse_gnn attribute of type ReverseGnnConfig with enabled=False."""
    cfg = ModelConfig()
    assert hasattr(cfg, "reverse_gnn")
    assert isinstance(cfg.reverse_gnn, ReverseGnnConfig)
    assert cfg.reverse_gnn.enabled is False


# ---------------------------------------------------------------------------
# HashiModelConfig.from_dict tests
# ---------------------------------------------------------------------------

def test_from_dict_parses_reasoning_and_reverse_gnn():
    """Test 8: from_dict with reasoning and reverse_gnn sub-dicts parses without TypeError."""
    config_dict = {
        "data": {},
        "model": {
            "reasoning": {"enabled": True, "steps": 3},
            "reverse_gnn": {"enabled": True, "separate_weights": False},
        },
        "training": {},
    }
    cfg = HashiModelConfig.from_dict(config_dict)
    assert cfg.model.reasoning.enabled is True
    assert cfg.model.reasoning.steps == 3
    assert cfg.model.reverse_gnn.enabled is True
    assert cfg.model.reverse_gnn.separate_weights is False


def test_from_dict_uses_defaults_when_subkeys_absent():
    """Test 9: from_dict without reasoning/reverse_gnn uses defaults (enabled=False for both)."""
    config_dict = {
        "data": {},
        "model": {},
        "training": {},
    }
    cfg = HashiModelConfig.from_dict(config_dict)
    assert cfg.model.reasoning.enabled is False
    assert cfg.model.reverse_gnn.enabled is False


def test_existing_yaml_configs_load_without_error():
    """Test 10: Existing YAML configs load without error via HashiModelConfig.from_yaml."""
    for path in [
        "configs/diffusion_solver_continuous.yaml",
        "configs/diffusion_solver_continuous_bptt.yaml",
    ]:
        cfg = HashiModelConfig.from_yaml(path)
        assert cfg is not None, f"Failed to load {path}"
