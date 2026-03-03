"""Tests for seeded val subset sampling (Step 1: val_sampler_seed + randperm fix)."""

import torch

from hashi_puzzle_solver.trainers.base import BaseTrainer


class _DummyTrainer(BaseTrainer):
    def run_epoch(self, loader, training=True, epoch=1, total_epochs=1):
        raise NotImplementedError


class _DummyDataset:
    def __init__(self, *args, **kwargs) -> None:
        self._size = 120

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int):
        return idx


def _base_config(val_sampler_seed: int = 42) -> dict:
    return {
        "data": {
            "root_dir": "dataset/",
            "limit": 20,
            "val_sampler_seed": val_sampler_seed,
        },
        "model": {
            "use_global_meta_node": False,
            "use_row_col_meta": False,
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 4,
            "epochs": 1,
            "num_workers": 0,
            "use_persistent_workers": False,
        },
    }


def test_val_sampler_uses_seeded_randperm(monkeypatch):
    """Val sampler indices should be a seeded random permutation, not sequential."""
    monkeypatch.setattr("hashi_puzzle_solver.trainers.base.HashiDataset", _DummyDataset)

    trainer = _DummyTrainer(_base_config(val_sampler_seed=7), torch.device("cpu"))
    loader = trainer.create_dataloader(split="val")

    assert isinstance(loader.sampler, torch.utils.data.SubsetRandomSampler)
    assert len(loader.sampler.indices) == 20
    # Indices must not be the trivial sequential range [0, 1, ..., 19]
    assert loader.sampler.indices != list(range(20))


def test_val_sampler_is_reproducible_across_calls(monkeypatch):
    """Same seed must produce identical index subsets on repeated calls."""
    monkeypatch.setattr("hashi_puzzle_solver.trainers.base.HashiDataset", _DummyDataset)

    trainer = _DummyTrainer(_base_config(val_sampler_seed=7), torch.device("cpu"))
    loader_a = trainer.create_dataloader(split="val")
    loader_b = trainer.create_dataloader(split="val")

    assert loader_a.sampler.indices == loader_b.sampler.indices


def test_val_sampler_differs_across_seeds(monkeypatch):
    """Different seeds must produce different index subsets."""
    monkeypatch.setattr("hashi_puzzle_solver.trainers.base.HashiDataset", _DummyDataset)

    trainer_a = _DummyTrainer(_base_config(val_sampler_seed=7), torch.device("cpu"))
    trainer_b = _DummyTrainer(_base_config(val_sampler_seed=123), torch.device("cpu"))

    loader_a = trainer_a.create_dataloader(split="val")
    loader_b = trainer_b.create_dataloader(split="val")

    assert loader_a.sampler.indices != loader_b.sampler.indices


def test_val_sampler_respects_dataset_size(monkeypatch):
    """Sampled indices must all be within [0, len(dataset))."""
    monkeypatch.setattr("hashi_puzzle_solver.trainers.base.HashiDataset", _DummyDataset)

    trainer = _DummyTrainer(_base_config(val_sampler_seed=42), torch.device("cpu"))
    loader = trainer.create_dataloader(split="val")

    dataset_size = 120  # matches _DummyDataset
    assert all(0 <= idx < dataset_size for idx in loader.sampler.indices)
