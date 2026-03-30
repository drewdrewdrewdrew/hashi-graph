"""Test callbacks for residual mode."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from .callbacks import MLflowCallback, PrintMetricsCallback
from .trainers.base import EpochMetrics


def test_mlflow_logs_residual_mse():
    """Test that MLflowCallback logs residual_mse metrics."""
    callback = MLflowCallback(
        experiment_name="test",
        run_name="test_run",
        params={},
    )
    callback._disabled = True
    
    trainer = MagicMock()
    trainer.config = {"training": {"mode": "residual"}}
    
    train_metrics = EpochMetrics()
    train_metrics.loss = 0.5
    train_metrics.residual_mse = 0.123
    
    val_metrics = EpochMetrics()
    val_metrics.loss = 0.6
    val_metrics.residual_mse = 0.234
    
    callback.on_epoch_end(trainer, epoch=1, train_metrics=train_metrics, val_metrics=val_metrics, full_rollout_metrics=None)


def test_print_metrics_residual_mode(capsys):
    """Test that PrintMetricsCallback shows ResMSE for residual mode."""
    callback = PrintMetricsCallback()
    
    trainer = MagicMock()
    trainer.config = {
        "training": {"mode": "residual"},
        "model": {"use_verification_head": False},
    }
    trainer.current_masking_rate = 1.0
    
    train_metrics = EpochMetrics()
    train_metrics.loss = 0.5
    train_metrics.residual_mse = 0.123
    train_metrics.ce_loss = 0.0
    train_metrics.degree_loss = 0.01
    train_metrics.crossing_loss = 0.02
    train_metrics.accuracy = 0.95
    train_metrics.perfect_accuracy = 0.80
    
    callback.on_epoch_end(
        trainer,
        epoch=1,
        train_metrics=train_metrics,
        val_metrics=None,
        full_rollout_metrics=None,
    )
    
    captured = capsys.readouterr()
    assert "ResMSE" in captured.out
    assert "residual" in captured.out.lower()
    assert "NoiseL" not in captured.out


def test_print_metrics_non_residual_mode(capsys):
    """Test that PrintMetricsCallback shows CE for non-residual modes."""
    callback = PrintMetricsCallback()
    
    trainer = MagicMock()
    trainer.config = {
        "training": {"mode": "diff-cont"},
        "model": {"use_verification_head": False},
    }
    trainer.current_masking_rate = 1.0
    
    train_metrics = EpochMetrics()
    train_metrics.loss = 0.5
    train_metrics.ce_loss = 0.456
    train_metrics.residual_mse = 0.0
    train_metrics.degree_loss = 0.01
    train_metrics.crossing_loss = 0.02
    train_metrics.noise_loss = 0.03
    train_metrics.accuracy = 0.95
    train_metrics.perfect_accuracy = 0.80
    
    callback.on_epoch_end(
        trainer,
        epoch=1,
        train_metrics=train_metrics,
        val_metrics=None,
        full_rollout_metrics=None,
    )
    
    captured = capsys.readouterr()
    assert "CE" in captured.out
    assert "ResMSE" not in captured.out
    assert "NoiseL" in captured.out


def test_print_metrics_flow_blind_mode(capsys):
    """Test that PrintMetricsCallback hides NoiseL for flow-blind mode."""
    callback = PrintMetricsCallback()
    
    trainer = MagicMock()
    trainer.config = {
        "training": {"mode": "flow-blind"},
        "model": {"use_verification_head": False},
    }
    trainer.current_masking_rate = 1.0
    
    train_metrics = EpochMetrics()
    train_metrics.loss = 0.5
    train_metrics.ce_loss = 0.456
    train_metrics.residual_mse = 0.0
    train_metrics.degree_loss = 0.01
    train_metrics.crossing_loss = 0.02
    train_metrics.noise_loss = 0.0
    train_metrics.accuracy = 0.95
    train_metrics.perfect_accuracy = 0.80
    
    callback.on_epoch_end(
        trainer,
        epoch=1,
        train_metrics=train_metrics,
        val_metrics=None,
        full_rollout_metrics=None,
    )
    
    captured = capsys.readouterr()
    assert "CE" in captured.out
    assert "NoiseL" not in captured.out
    assert "ResMSE" not in captured.out
