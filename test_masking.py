#!/usr/bin/env python3
"""
Test script for masking rate calculation
"""

import math
from typing import Dict, Any

def get_masking_rate(epoch: int, masking_config: Dict[str, Any], total_epochs: int) -> float:
    """
    Calculate progressive masking rate based on epoch, including warmup and cooldown periods.
    """
    if not masking_config.get('enabled', False):
        return 0.0

    start_rate = masking_config.get('start_rate', 0.0)
    warmup_epochs = masking_config.get('warmup_epochs', 0)
    cooldown_epochs = masking_config.get('cooldown_epochs', 0)

    # Calculate effective epochs for ramping up masking
    rampup_epochs = total_epochs - warmup_epochs - cooldown_epochs
    if rampup_epochs <= 0:
        # If no rampup period, masking is either start_rate or end_rate depending on config
        if epoch <= warmup_epochs:
            return start_rate
        else:
            return masking_config.get('end_rate', 1.0) # Assume full masking if no rampup

    # Handle warmup phase
    if epoch <= warmup_epochs:
        return start_rate

    # Handle cooldown phase (after rampup, before end of training)
    if epoch > (warmup_epochs + rampup_epochs):
        return masking_config.get('end_rate', 1.0) # Maintain full masking during cooldown

    # Calculate progress during rampup phase
    end_rate = masking_config.get('end_rate', 1.0)
    schedule = masking_config.get('schedule', 'cosine')

    # Normalize epoch to the rampup period
    progress = (epoch - warmup_epochs) / rampup_epochs
    progress = min(progress, 1.0)  # Ensure it doesn't exceed 1.0

    if schedule == 'cosine':
        rate = start_rate + (end_rate - start_rate) * (1 - math.cos(math.pi * progress)) / 2
    elif schedule == 'linear':
        rate = start_rate + (end_rate - start_rate) * progress
    elif schedule == 'constant':
        rate = start_rate
    else:
        raise ValueError(f"Unknown masking schedule: {schedule}. Choose 'cosine', 'linear', or 'constant'.")

    return float(rate)

# Test with the config from masking_experiment.yaml
masking_config = {
    'enabled': True,
    'start_rate': 0.75,
    'end_rate': 1.0,
    'schedule': 'cosine',
    'warmup_epochs': 0,
    'cooldown_epochs': 25
}

total_epochs = 50

print('Masking rates for first 10 epochs:')
for epoch in range(1, 11):
    rate = get_masking_rate(epoch, masking_config, total_epochs)
    print('.3f')

print('\nMasking rates for epochs 40-50:')
for epoch in range(40, 51):
    rate = get_masking_rate(epoch, masking_config, total_epochs)
    print('.3f')

print('\n' + '='*50)
print('TESTING WARMUP BEHAVIOR')
print('='*50)

# Test with warmup
masking_config_warmup = {
    'enabled': True,
    'start_rate': 0.75,
    'end_rate': 1.0,
    'schedule': 'cosine',
    'warmup_epochs': 5,
    'cooldown_epochs': 20
}

total_epochs_warmup = 50
rampup_epochs = total_epochs_warmup - 5 - 20  # 25

print(f'\nWith warmup_epochs=5, rampup starts at epoch 6')
print(f'Rampup period: epochs 6-{5+rampup_epochs} (epochs 6-30)')
print(f'Cooldown: epochs 31-50')

print('\nFirst 10 epochs with warmup:')
for epoch in range(1, 11):
    rate = get_masking_rate(epoch, masking_config_warmup, total_epochs_warmup)
    print('.3f')
