"""Test masking rate calculation functionality."""

import pytest

from hashi_puzzle_solver.engine import get_masking_rate


class TestMaskingRateCalculation:
    """Test masking rate calculation with different schedules and configurations."""

    def test_cosine_schedule_no_warmup(self) -> None:
        """Test cosine schedule without warmup or cooldown."""
        config = {
            "enabled": True,
            "start_rate": 0.75,
            "end_rate": 1.0,
            "schedule": "cosine",
            "warmup_epochs": 0,
            "cooldown_epochs": 25,
        }
        total_epochs = 50

        # Test first few epochs
        assert (
            get_masking_rate(1, config, total_epochs) > 0.75
        )  # Should start above 0.75
        assert get_masking_rate(5, config, total_epochs) > 0.75
        assert get_masking_rate(10, config, total_epochs) > 0.8

        # Test later epochs (cooldown)
        assert (
            get_masking_rate(40, config, total_epochs) == 1.0
        )  # Full masking in cooldown
        assert get_masking_rate(45, config, total_epochs) == 1.0
        assert get_masking_rate(50, config, total_epochs) == 1.0

    def test_warmup_behavior(self) -> None:
        """Test masking rate with warmup period."""
        config = {
            "enabled": True,
            "start_rate": 0.75,
            "end_rate": 1.0,
            "schedule": "cosine",
            "warmup_epochs": 5,
            "cooldown_epochs": 20,
        }
        total_epochs = 50

        # Warmup period: should maintain start_rate
        assert get_masking_rate(1, config, total_epochs) == 0.75
        assert get_masking_rate(3, config, total_epochs) == 0.75
        assert get_masking_rate(5, config, total_epochs) == 0.75

        # Rampup starts at epoch 6
        assert get_masking_rate(6, config, total_epochs) > 0.75

        # Cooldown period (epochs 31-50)
        assert get_masking_rate(31, config, total_epochs) == 1.0
        assert get_masking_rate(40, config, total_epochs) == 1.0
        assert get_masking_rate(50, config, total_epochs) == 1.0

    def test_linear_schedule(self) -> None:
        """Test linear masking schedule."""
        config = {
            "enabled": True,
            "start_rate": 0.0,
            "end_rate": 1.0,
            "schedule": "linear",
            "warmup_epochs": 10,
            "cooldown_epochs": 10,
        }
        total_epochs = 100

        # Warmup
        assert get_masking_rate(5, config, total_epochs) == 0.0

        # Linear progression: (epoch-10)/(100-10-10) = (epoch-10)/80
        # At epoch 50: (50-10)/80 = 40/80 = 0.5
        assert get_masking_rate(50, config, total_epochs) == 0.5

        # Cooldown
        assert get_masking_rate(95, config, total_epochs) == 1.0

    def test_constant_schedule(self) -> None:
        """Test constant masking schedule."""
        config = {
            "enabled": True,
            "start_rate": 0.5,
            "end_rate": 1.0,
            "schedule": "constant",
            "warmup_epochs": 5,
            "cooldown_epochs": 5,
        }
        total_epochs = 50

        # Should maintain start_rate throughout
        assert get_masking_rate(10, config, total_epochs) == 0.5
        assert get_masking_rate(25, config, total_epochs) == 0.5
        assert get_masking_rate(40, config, total_epochs) == 0.5

    def test_disabled_masking(self) -> None:
        """Test disabled masking returns 0.0."""
        config = {"enabled": False}
        assert get_masking_rate(25, config, 100) == 0.0

    def test_invalid_schedule(self) -> None:
        """Test invalid schedule raises ValueError."""
        config = {
            "enabled": True,
            "schedule": "invalid",
            "start_rate": 0.0,
            "end_rate": 1.0,
        }

        with pytest.raises(ValueError, match="Unknown masking schedule"):
            get_masking_rate(25, config, 100)
