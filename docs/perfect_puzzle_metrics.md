# Perfect Puzzle Accuracy Metrics

This document describes the new perfect puzzle accuracy metrics added to the training and evaluation pipeline.

## Overview

In addition to edge-level accuracy, the training system now tracks **perfect puzzle accuracy** - the percentage of puzzles where all edges are correctly predicted. This is a more stringent metric that better reflects the model's ability to fully solve puzzles.

## What Changed

### 1. New `train_utils.py` Module

Contains utility functions for calculating perfect puzzle metrics at both single-puzzle and batch levels:

**Single-Puzzle Functions** (work directly with dataset):
- `is_puzzle_perfect()`: Check if a single puzzle is perfectly solved
- `evaluate_puzzle()`: Comprehensive evaluation of a single puzzle

**Batch-Level Functions** (for efficient training):
- `calculate_perfect_puzzle_accuracy()`: Core function to check if all edges in each puzzle are correct
- `calculate_batch_perfect_puzzles()`: Wrapper for batch processing
- `aggregate_perfect_puzzle_stats()`: Aggregates stats across multiple batches
- `get_edge_batch_indices()`: Helper to map edges to their puzzle indices

### 2. Updated Training Script (`train.py`)

- The training loop now includes a configurable `eval_interval` parameter (default: 1) to control how frequently validation is performed.
- The best model (based on validation loss) is now saved to the MLflow run's artifact directory (`mlruns/<run_id>/artifacts/best_model.pt`).
- At the end of the run, the best model is reloaded and logged as an MLflow artifact.

The training and evaluation functions now return three values instead of two:

```python
# Before
train_loss, train_acc = train_epoch(...)
val_loss, val_acc = evaluate(...)

# After
train_loss, train_acc, train_perfect = train_epoch(...)
val_loss, val_acc, val_perfect = evaluate(...)
```

The console output now includes perfect puzzle accuracy (and only for epochs where evaluation is performed):

```
Epoch: 001, Train Mask: 0.00, Train Loss: 0.4811, Train Acc: 0.7879, Train Perfect: 0.1250, 
Val Loss: 4.8005, Val Acc: 0.4432, Val Perfect: 0.0523
```

MLflow now logs these additional metrics:
- `train_perfect_puzzle_acc`
- `val_perfect_puzzle_acc`
- `best_val_loss`
- The best model as an artifact (`model`)

### 3. New Evaluation Script (`evaluate_model.py`)

A standalone script to evaluate any trained model on any dataset.

## Usage

### During Training

Perfect puzzle metrics are automatically calculated and logged during training. No changes needed to your training commands:

```bash
uv run -m src.train --config configs/masking_experiment.yaml
```

Check MLflow UI to see the new metrics:
```bash
mlflow ui
```

### Standalone Evaluation

Evaluate a trained model on any dataset split. The script supports two modes:

**Puzzle-by-Puzzle Mode** (default, cleaner, works directly with dataset):
```bash
# Evaluate on validation set (puzzle-by-puzzle)
python -m src.evaluate_model \
    --model_path path/to/best_model.pt \
    --config configs/masking_experiment.yaml \
    --split val

# Evaluate on test set
python -m src.evaluate_model \
    --model_path path/to/best_model.pt \
    --config configs/masking_experiment.yaml \
    --split test \
    --mode puzzle
```

**Batched Mode** (faster for large datasets):
```bash
# Evaluate using batched processing
python -m src.evaluate_model \
    --model_path path/to/best_model.pt \
    --config configs/masking_experiment.yaml \
    --split val \
    --mode batch \
    --batch_size 128
```

### Example Output

```
============================================================
EVALUATION RESULTS
============================================================
Dataset Split:           val
Total Puzzles:           4500
Total Edges:             45892
------------------------------------------------------------
Loss:                    0.4932
Edge Accuracy:           0.7362 (33785/45892)
Perfect Puzzle Accuracy: 0.1847 (831/4500)
------------------------------------------------------------
Per-Class Accuracy:
  Class 0: 0.8234 (28934 edges)
  Class 1: 0.6521 (12456 edges)
  Class 2: 0.5892 (4502 edges)
============================================================
```

## Understanding the Metrics

### Edge Accuracy
- Percentage of individual edges correctly predicted
- More lenient metric
- Example: 0.7362 means 73.62% of all edges are correct

### Perfect Puzzle Accuracy
- Percentage of puzzles where ALL edges are correct
- More stringent metric
- Better reflects true puzzle-solving ability
- Example: 0.1847 means 18.47% of puzzles are perfectly solved

### Why Both Matter

- **Edge Accuracy**: Shows the model is learning edge patterns
- **Perfect Puzzle Accuracy**: Shows the model can solve complete puzzles
- A model might have 90% edge accuracy but only 10% perfect puzzle accuracy if errors are distributed across many puzzles

## Implementation Notes

### Two Evaluation Approaches

**1. Puzzle-by-Puzzle (Dataset-Level)**
- Works directly with the `HashiDataset` object
- Evaluates one puzzle at a time using `evaluate_puzzle()`
- Cleaner and easier to understand
- Ideal for analysis and debugging
- Recommended for auxiliary evaluation script

**2. Batched (Loader-Level)**
- Works with PyTorch Geometric `DataLoader` batches
- More efficient for large datasets
- Uses batch indices to separate puzzles
- Required for training (gradient computation needs batching)
- Recommended for training loop

### Memory Efficiency
The perfect puzzle calculation is done per-batch (or per-puzzle) and aggregated, minimizing memory overhead.

### Masking Consideration
Perfect puzzle accuracy is calculated on the masked/visible edges during training, but validation always uses 100% masking (true puzzle solving from scratch).

### Integration with Dataset
The new `evaluate_puzzle()` function leverages the fact that each puzzle is a separate `Data` object in the dataset, making single-puzzle evaluation straightforward without needing to reconstruct puzzle boundaries from batch indices.

## API Reference

### `calculate_perfect_puzzle_accuracy(predictions, targets, edge_masks, batch_indices)`

Calculate perfect puzzle accuracy.

**Args:**
- `predictions` (Tensor): Predicted edge labels [num_edges]
- `targets` (Tensor): Ground truth labels [num_edges]
- `edge_masks` (Tensor): Boolean mask for original edges [num_edges]
- `batch_indices` (Tensor): Batch index for each edge [num_edges]

**Returns:**
- `perfect_accuracy` (float): Fraction of perfect puzzles (0.0 to 1.0)
- `perfect_puzzles` (int): Number of perfectly solved puzzles
- `num_puzzles` (int): Total number of puzzles

### `evaluate_model(model, loader, device, verbose=True)`

Comprehensive evaluation function.

**Returns:** Dict with keys:
- `loss`: Average cross-entropy loss
- `edge_accuracy`: Fraction of correct edges
- `perfect_puzzle_accuracy`: Fraction of perfect puzzles
- `total_puzzles`, `perfect_puzzles`: Counts
- `total_edges`, `correct_edges`: Edge counts
- `per_class_accuracy`: List of accuracy per class
- `class_total`: List of edge counts per class

## Future Enhancements

Potential additions:
- Partial puzzle accuracy (e.g., 90%+ correct)
- Per-difficulty perfect accuracy tracking
- Confidence-weighted metrics
- Time-to-solve tracking for perfect puzzles

