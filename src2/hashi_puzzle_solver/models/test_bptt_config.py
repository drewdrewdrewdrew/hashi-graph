"""Tests for BpttConfig dataclass and its integration into TrainingConfig/HashiModelConfig."""

import pytest

from src2.hashi_puzzle_solver.models.config import (
    BpttConfig,
    HashiModelConfig,
    TrainingConfig,
)


class TestBpttConfigDefaults:
    """BpttConfig() with no args produces correct defaults."""

    def test_default_enabled_is_false(self):
        cfg = BpttConfig()
        assert cfg.enabled is False

    def test_default_window_is_8(self):
        cfg = BpttConfig()
        assert cfg.window == 8

    def test_default_stride_is_4(self):
        cfg = BpttConfig()
        assert cfg.stride == 4

    def test_default_loss_ema_decay_is_0_9(self):
        cfg = BpttConfig()
        assert cfg.loss_ema_decay == 0.9


class TestBpttConfigValidation:
    """BpttConfig raises ValueError for out-of-range inputs."""

    def test_window_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="bptt.window must be >= 1"):
            BpttConfig(window=0)

    def test_window_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="bptt.window must be >= 1"):
            BpttConfig(window=-3)

    def test_stride_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="bptt.stride must be >= 1"):
            BpttConfig(stride=0)

    def test_stride_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="bptt.stride must be >= 1"):
            BpttConfig(stride=-1)

    def test_loss_ema_decay_equals_one_raises_value_error(self):
        with pytest.raises(ValueError, match="bptt.loss_ema_decay must be in"):
            BpttConfig(loss_ema_decay=1.0)

    def test_loss_ema_decay_above_one_raises_value_error(self):
        with pytest.raises(ValueError, match="bptt.loss_ema_decay must be in"):
            BpttConfig(loss_ema_decay=1.5)

    def test_loss_ema_decay_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="bptt.loss_ema_decay must be in"):
            BpttConfig(loss_ema_decay=-0.1)

    def test_valid_values_do_not_raise(self):
        cfg = BpttConfig(enabled=True, window=4, stride=2, loss_ema_decay=0.95)
        assert cfg.enabled is True
        assert cfg.window == 4
        assert cfg.stride == 2
        assert cfg.loss_ema_decay == 0.95

    def test_loss_ema_decay_zero_is_valid(self):
        cfg = BpttConfig(loss_ema_decay=0.0)
        assert cfg.loss_ema_decay == 0.0

    def test_window_one_is_valid(self):
        cfg = BpttConfig(window=1)
        assert cfg.window == 1

    def test_stride_one_is_valid(self):
        cfg = BpttConfig(stride=1)
        assert cfg.stride == 1


class TestTrainingConfigBpttField:
    """TrainingConfig has a typed bptt field."""

    def test_training_config_has_bptt_attribute(self):
        training = TrainingConfig()
        assert hasattr(training, "bptt")

    def test_training_config_bptt_is_bptt_config_instance(self):
        training = TrainingConfig()
        assert isinstance(training.bptt, BpttConfig)

    def test_training_config_bptt_defaults_match(self):
        training = TrainingConfig()
        assert training.bptt.enabled is False
        assert training.bptt.window == 8
        assert training.bptt.stride == 4
        assert training.bptt.loss_ema_decay == 0.9


class TestHashiModelConfigFromDict:
    """HashiModelConfig.from_dict correctly handles bptt field."""

    def test_from_empty_dict_produces_bptt_with_defaults(self):
        cfg = HashiModelConfig.from_dict({})
        assert cfg.training.bptt.enabled is False
        assert cfg.training.bptt.window == 8
        assert cfg.training.bptt.stride == 4
        assert cfg.training.bptt.loss_ema_decay == 0.9

    def test_from_dict_no_bptt_key_no_error(self):
        cfg = HashiModelConfig.from_dict({"training": {"learning_rate": 0.001}})
        assert cfg.training.bptt.enabled is False

    def test_from_dict_bptt_enabled_and_window(self):
        cfg = HashiModelConfig.from_dict(
            {"training": {"bptt": {"enabled": True, "window": 4}}}
        )
        assert cfg.training.bptt.enabled is True
        assert cfg.training.bptt.window == 4
        assert cfg.training.bptt.stride == 4  # default

    def test_from_dict_bptt_all_fields(self):
        cfg = HashiModelConfig.from_dict(
            {
                "training": {
                    "bptt": {
                        "enabled": True,
                        "window": 16,
                        "stride": 8,
                        "loss_ema_decay": 0.95,
                    }
                }
            }
        )
        assert cfg.training.bptt.enabled is True
        assert cfg.training.bptt.window == 16
        assert cfg.training.bptt.stride == 8
        assert cfg.training.bptt.loss_ema_decay == 0.95

    def test_from_dict_existing_training_fields_not_broken(self):
        cfg = HashiModelConfig.from_dict(
            {"training": {"learning_rate": 0.001, "batch_size": 64}}
        )
        assert cfg.training.learning_rate == 0.001
        assert cfg.training.batch_size == 64
        assert cfg.training.bptt.enabled is False
