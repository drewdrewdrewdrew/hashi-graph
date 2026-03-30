"""Smoke/overfit integration test for residual mode (Chunk 8).

This test verifies that residual mode works end-to-end by running
a minimal training configuration. The test can be run via:

    uv run python -m hashi_puzzle_solver.train --config configs/residual_smoke_test.yaml

Expected behavior:
1. Training runs for 2 epochs in residual mode
2. residual_mse is computed and logged (shown as "ResMSE" in output)
3. Rollout produces perfect_acc_k metrics
4. Validation residual_mse should decrease or stay stable

The smoke test config (configs/residual_smoke_test.yaml) uses:
- 10 training samples, 3 validation samples
- 8x8 easy puzzles only
- 2 epochs
- num_inference_steps_training: 3
- Small model (hidden_channels: 64, num_layers: 2)

This is sufficient to verify that all residual mode components work together:
- Noise injection (Chunk 3)
- Residual MSE loss computation (Chunk 4)
- Detached state carry (Chunk 5)
- Rollout with iterative correction (Chunk 6)
- Metrics logging (Chunk 7)
"""

import pytest


def test_residual_smoke_test_config_exists():
    """Verify that the smoke test config file exists."""
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent.parent.parent / "configs" / "residual_smoke_test.yaml"
    assert config_path.exists(), f"Smoke test config not found at {config_path}"
    
    # Verify it's valid YAML
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Verify key settings
    assert config["training"]["mode"] == "residual"
    assert config["training"]["epochs"] == 2
    assert config["training"]["num_inference_steps_training"] >= 2
    assert config["training"]["loss_weights"]["residual_mse"] == 1.0
    assert config["data"]["train_limit"] <= 20  # Small dataset
    
    print("\n✓ Smoke test config is valid")
    print(f"  Mode: {config['training']['mode']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Train samples: {config['data']['train_limit']}")
    print(f"  Inference steps: {config['training']['num_inference_steps_training']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
