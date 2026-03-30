"""Test that residual mode is correctly routed to DiffusionTrainer."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from hashi_puzzle_solver.engine import create_trainer
from hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer


@patch('hashi_puzzle_solver.engine.DiffusionTrainer')
def test_residual_mode_routes_to_diffusion_trainer(mock_diffusion_trainer):
    """Verify that training.mode: residual creates a DiffusionTrainer instance."""
    mock_instance = MagicMock(spec=DiffusionTrainer)
    mock_diffusion_trainer.return_value = mock_instance
    
    config = {
        "training": {
            "mode": "residual",
        },
    }
    
    device = torch.device("cpu")
    trainer = create_trainer(config, device, callbacks=None)
    
    # Verify DiffusionTrainer was called
    mock_diffusion_trainer.assert_called_once_with(config, device, None)
    assert trainer is mock_instance


@patch('hashi_puzzle_solver.engine.DiffusionTrainer')
def test_residual_mode_case_insensitive(mock_diffusion_trainer):
    """Verify that mode matching is case-insensitive."""
    mock_instance = MagicMock(spec=DiffusionTrainer)
    mock_diffusion_trainer.return_value = mock_instance
    
    config = {
        "training": {
            "mode": "RESIDUAL",
        },
    }
    
    device = torch.device("cpu")
    trainer = create_trainer(config, device, callbacks=None)
    
    # Verify DiffusionTrainer was called
    mock_diffusion_trainer.assert_called_once_with(config, device, None)
    assert trainer is mock_instance


@patch('hashi_puzzle_solver.engine.OneShotTrainer')
@patch('hashi_puzzle_solver.engine.DiffusionTrainer')
def test_other_modes_not_affected(mock_diffusion_trainer, mock_oneshot_trainer):
    """Verify that adding residual mode doesn't affect other mode routing."""
    mock_oneshot_instance = MagicMock()
    mock_oneshot_trainer.return_value = mock_oneshot_instance
    
    config = {
        "training": {
            "mode": "one-shot",
        },
    }
    
    device = torch.device("cpu")
    trainer = create_trainer(config, device, callbacks=None)
    
    # Verify OneShotTrainer was called, not DiffusionTrainer
    mock_oneshot_trainer.assert_called_once()
    mock_diffusion_trainer.assert_not_called()
    assert trainer is mock_oneshot_instance
