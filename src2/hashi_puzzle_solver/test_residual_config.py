"""
Test for residual_solver.yaml configuration file.
Verifies that the config loads correctly and contains all required keys.
"""

import yaml
from pathlib import Path


def test_residual_config_loads():
    """Test that residual_solver.yaml loads and has required keys."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "residual_solver.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Verify top-level structure
    assert "training" in config, "Config missing 'training' section"
    assert "model" in config, "Config missing 'model' section"
    assert "data" in config, "Config missing 'data' section"
    
    training = config["training"]
    
    # Verify residual mode is set
    assert training["mode"] == "residual", f"Expected mode='residual', got '{training['mode']}'"
    
    # Verify required training keys
    assert "num_inference_steps_training" in training, "Missing num_inference_steps_training"
    assert "sigma_max" in training, "Missing sigma_max"
    assert "scale_max" in training, "Missing scale_max"
    assert "rollout_init" in training, "Missing rollout_init"
    
    # Verify loss_weights structure
    assert "loss_weights" in training, "Missing loss_weights"
    loss_weights = training["loss_weights"]
    assert "residual_mse" in loss_weights, "Missing loss_weights.residual_mse"
    
    # Verify residual_mse has non-zero weight
    assert loss_weights["residual_mse"] > 0, "residual_mse weight should be > 0"
    
    # Verify auxiliary losses are present (even if zero)
    assert "degree" in loss_weights, "Missing loss_weights.degree"
    assert "crossing" in loss_weights, "Missing loss_weights.crossing"
    assert "verify" in loss_weights, "Missing loss_weights.verify"
    
    # Verify model flags appropriate for residual mode
    model = config["model"]
    assert model["use_continuous_edge_labels"] is True, "use_continuous_edge_labels must be True for residual mode"
    assert model["use_noise_head"] is False, "use_noise_head should be False for residual mode"
    assert model["use_component_meta"] is False, "use_component_meta should be False for residual mode v1"
    
    # Verify BPTT is disabled
    assert "bptt" in training, "Missing bptt section"
    assert training["bptt"]["enabled"] is False, "BPTT must be disabled for residual mode"
    
    # Verify rollout_init is valid
    assert training["rollout_init"] in ["noise", "zeros"], f"rollout_init must be 'noise' or 'zeros', got '{training['rollout_init']}'"


def test_residual_config_values():
    """Test that key configuration values are sensible."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "residual_solver.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    training = config["training"]
    
    # Check numeric ranges
    assert training["sigma_max"] > 0, "sigma_max must be positive"
    assert training["scale_min"] > 0, "scale_min must be positive"
    assert training["scale_max"] > training["scale_min"], "scale_max must be > scale_min"
    assert 0 <= training["alpha_power"] <= 10, "alpha_power should be in reasonable range"
    assert 0 <= training["zero_signal_prob"] <= 1, "zero_signal_prob must be a probability"
    assert training["num_inference_steps_training"] >= 1, "num_inference_steps_training must be >= 1"
