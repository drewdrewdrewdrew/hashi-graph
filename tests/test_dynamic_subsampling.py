import os
import json
import torch
import pytest
import shutil
from pathlib import Path
from hashi_puzzle_solver.engine import Trainer

@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dummy dataset with 10 train and 10 val puzzles."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val"]:
        for i in range(10):
            puzzle = {
                "split": split,
                "generation_params": {"size": 8, "difficulty": 0},
                "graph": {
                    "nodes": [
                        {"id": 0, "pos": [0, 0], "n": 1}, 
                        {"id": 1, "pos": [0, 1], "n": 1}
                    ],
                    "edges": [{"source": 0, "target": 1, "label": 1}]
                }
            }
            with open(raw_dir / f"puzzle_{split}_{i}.json", "w") as f:
                json.dump(puzzle, f)
    
    return tmp_path

def test_dynamic_subsampling_logic(temp_dataset):
    """Verify that 'limit' in config correctly implements dynamic subsampling."""
    config = {
        "data": {
            "root_dir": str(temp_dataset),
            "limit": 3,
            "size": [8],
            "difficulty": 0
        },
        "model": {
            "use_degree": False,
            "use_global_meta_node": False,
            "use_row_col_meta": False,
            "use_distance": False,
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_conflict_status": True
        },
        "training": {
            "batch_size": 1,
            "learning_rate": 0.001
        }
    }

    trainer = Trainer(config, torch.device("cpu"))
    
    # Test Train Loader
    train_loader = trainer.create_dataloader(split="train")
    # limit=3 means 3 batches if batch_size=1
    assert len(train_loader) == 3
    
    # Test Val Loader
    val_loader = trainer.create_dataloader(split="val")
    # val should also be limited to 3
    assert len(val_loader) == 3
    
    # Verify dataset actually knows about all 10 files
    # This proves we didn't use a hard limit at the dataset level
    assert len(train_loader.dataset) == 10
    assert len(val_loader.dataset) == 10

def test_no_subsampling_logic(temp_dataset):
    """Verify that if limit is None, all data is used."""
    config = {
        "data": {
            "root_dir": str(temp_dataset),
            "limit": None,
            "size": [8],
            "difficulty": 0
        },
        "model": {
            "use_degree": False,
            "use_global_meta_node": False,
            "use_row_col_meta": False,
            "use_distance": False,
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_conflict_status": True
        },
        "training": {
            "batch_size": 1,
            "learning_rate": 0.001
        }
    }

    trainer = Trainer(config, torch.device("cpu"))
    
    train_loader = trainer.create_dataloader(split="train")
    assert len(train_loader) == 10
    
    val_loader = trainer.create_dataloader(split="val")
    assert len(val_loader) == 10
