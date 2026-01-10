"""Test masking rate calculation functionality."""

import pytest

from hashi_puzzle_solver.masking import MaskingStrategy


class TestMaskingRateCalculation:
    """Test masking rate calculation with different schedules and configurations."""

    def test_cosine_schedule_no_warmup(self) -> None:
        """Test cosine schedule without warmup or cooldown."""
        config = {
            "training": {
                "masking": {
                    "enabled": True,
                    "start_rate": 0.75,
                    "end_rate": 1.0,
                    "schedule": "cosine",
                    "warmup_epochs": 0,
                    "cooldown_epochs": 25,
                }
            }
        }
        strategy = MaskingStrategy(config)
        total_epochs = 50

        # Test first few epochs
        assert (
            strategy.get_rate(1, total_epochs) > 0.75
        )  # Should start above 0.75
        assert strategy.get_rate(5, total_epochs) > 0.75
        assert strategy.get_rate(10, total_epochs) > 0.8

        # Test later epochs (cooldown)
        assert (
            strategy.get_rate(40, total_epochs) == 1.0
        )  # Full masking in cooldown
        assert strategy.get_rate(45, total_epochs) == 1.0
        assert strategy.get_rate(50, total_epochs) == 1.0

    def test_warmup_behavior(self) -> None:
        """Test masking rate with warmup period."""
        config = {
            "training": {
                "masking": {
                    "enabled": True,
                    "start_rate": 0.75,
                    "end_rate": 1.0,
                    "schedule": "cosine",
                    "warmup_epochs": 5,
                    "cooldown_epochs": 20,
                }
            }
        }
        strategy = MaskingStrategy(config)
        total_epochs = 50

        # Warmup period: should maintain start_rate
        assert strategy.get_rate(1, total_epochs) == 0.75
        assert strategy.get_rate(3, total_epochs) == 0.75
        assert strategy.get_rate(5, total_epochs) == 0.75

        # Rampup starts at epoch 6
        assert strategy.get_rate(6, total_epochs) > 0.75

        # Cooldown period (epochs 31-50)
        assert strategy.get_rate(31, total_epochs) == 1.0
        assert strategy.get_rate(40, total_epochs) == 1.0
        assert strategy.get_rate(50, total_epochs) == 1.0

    def test_linear_schedule(self) -> None:
        """Test linear masking schedule."""
        config = {
            "training": {
                "masking": {
                    "enabled": True,
                    "start_rate": 0.0,
                    "end_rate": 1.0,
                    "schedule": "linear",
                    "warmup_epochs": 10,
                    "cooldown_epochs": 10,
                }
            }
        }
        strategy = MaskingStrategy(config)
        total_epochs = 100

        # Warmup
        assert strategy.get_rate(5, total_epochs) == 0.0

        # Linear progression: (epoch-10)/(100-10-10) = (epoch-10)/80
        # At epoch 50: (50-10)/80 = 40/80 = 0.5
        assert strategy.get_rate(50, total_epochs) == 0.5

        # Cooldown
        assert strategy.get_rate(95, total_epochs) == 1.0

    def test_constant_schedule(self) -> None:
        """Test constant masking schedule."""
        config = {
            "training": {
                "masking": {
                    "enabled": True,
                    "start_rate": 0.5,
                    "end_rate": 1.0,
                    "schedule": "constant",
                    "warmup_epochs": 5,
                    "cooldown_epochs": 5,
                }
            }
        }
        strategy = MaskingStrategy(config)
        total_epochs = 50

        # Should maintain start_rate throughout
        assert strategy.get_rate(10, total_epochs) == 0.5
        assert strategy.get_rate(25, total_epochs) == 0.5
        assert strategy.get_rate(40, total_epochs) == 0.5

    def test_disabled_masking(self) -> None:
        """Test disabled masking returns 0.0."""
        config = {"training": {"masking": {"enabled": False}}}
        strategy = MaskingStrategy(config)
        assert strategy.get_rate(25, 100) == 0.0

    def test_invalid_schedule(self) -> None:
        """Test invalid schedule raises ValueError."""
        config = {
            "training": {
                "masking": {
                    "enabled": True,
                    "schedule": "invalid",
                    "start_rate": 0.0,
                    "end_rate": 1.0,
                }
            }
        }
        strategy = MaskingStrategy(config)

        with pytest.raises(ValueError, match="Unknown masking schedule"):
            strategy.get_rate(25, 100)
