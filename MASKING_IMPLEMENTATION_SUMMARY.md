# Progressive Edge Masking Implementation Summary

## Overview
Successfully implemented a progressive masking system (0% → 100%) for training the Hashi puzzle solver. The system enables the model to learn constraint propagation by gradually hiding edge labels during training, culminating in pure puzzle-solving capability.

## Key Features Implemented

### 1. **Toggleable Edge Labels as Features**
- Added `use_edge_labels_as_features` parameter to `HashiDataset`
- Edge features now support both:
  - **3D mode** (default): `[inv_dx, inv_dy, is_meta]`
  - **5D mode** (masking): `[inv_dx, inv_dy, is_meta, bridge_label, is_labeled]`

### 2. **Progressive Masking System**
- **`get_masking_rate()`**: Calculates masking rate based on epoch
  - Supports cosine, linear, and constant schedules
  - Configurable warmup period
  - Smooth progression from 0% to 100% masking
  
- **`apply_edge_label_masking()`**: Masks edge labels during training
  - Only masks original puzzle edges (not meta edges)
  - Zeros out `bridge_label` and `is_labeled` features
  - Respects masking rate for curriculum learning

### 3. **Model Updates**
All edge-aware models now support dynamic edge dimensions:
- **GATEdgeClassifier**: Added `edge_dim` parameter (default: 3)
- **GINEEdgeClassifier**: Added `edge_dim` parameter (default: 3)
- **TransformerEdgeClassifier**: Added `edge_dim` parameter (default: 3)
- **GCNEdgeClassifier**: No changes (doesn't use edge attributes)

### 4. **Training Pipeline Updates**
- `train_epoch()` now accepts `masking_rate` parameter
- Training loop calculates progressive masking rate per epoch
- MLflow logging includes `masking_rate` metric for tracking

## Configuration Files Created

### 1. `configs/transformer_solver.yaml`
- Transformer with 0% → 100% masking
- 6 layers, 4 heads, 64 embedding dim
- Cosine schedule, 20 epoch warmup
- **Purpose**: Full puzzle solving with global attention

### 2. `configs/gat_solver.yaml`
- GAT with 0% → 100% masking
- Same hyperparameters as Transformer
- **Purpose**: Compare local attention mechanism

### 3. `configs/gine_solver.yaml`
- GINE with 0% → 100% masking
- Same hyperparameters (no heads parameter)
- **Purpose**: Compare edge-conditioned message passing

### 4. `configs/baseline_no_masking.yaml`
- Transformer with masking disabled
- Labels always visible (100%)
- **Purpose**: Baseline for supervised learning comparison

## Backward Compatibility

✅ **Fully backward compatible** with existing code:
- `configs/base_config.yaml` works unchanged (3D edge features)
- Default `use_edge_labels_as_features=False` maintains original behavior
- Default `masking.enabled=False` disables masking
- All existing models continue to work

## Toggleable Parameters

### Model Configuration
```yaml
model:
  type: "gcn" | "gat" | "gine" | "transformer"
  use_degree: true | false
  use_meta_node: true | false
  use_row_col_meta: true | false
  use_distance: true | false
  use_edge_labels_as_features: true | false  # NEW
```

### Masking Configuration
```yaml
training:
  masking:
    enabled: true | false           # Master toggle
    schedule: "cosine" | "linear"   # Progression type
    start_rate: 0.0 - 1.0          # Starting masking %
    end_rate: 0.0 - 1.0            # Ending masking % (1.0 = 100%)
    warmup_epochs: 0 - N           # Epochs before masking begins
```

## Usage Examples

### Train Transformer with Full Masking
```bash
python -m src.train --config configs/transformer_solver.yaml
```

### Train GAT for Comparison
```bash
python -m src.train --config configs/gat_solver.yaml
```

### Train Baseline (No Masking)
```bash
python -m src.train --config configs/baseline_no_masking.yaml
```

### Use Existing Config (Backward Compatible)
```bash
python -m src.train --config configs/base_config.yaml
```

## Comparison Matrix

| Config | Model | Edge Dim | Masking | Purpose |
|--------|-------|----------|---------|---------|
| `base_config.yaml` | GAT | 3D | Disabled | Original baseline |
| `baseline_no_masking.yaml` | Transformer | 5D | Disabled | Supervised baseline |
| `transformer_solver.yaml` | Transformer | 5D | 0%→100% | Full puzzle solving |
| `gat_solver.yaml` | GAT | 5D | 0%→100% | Compare attention |
| `gine_solver.yaml` | GINE | 5D | 0%→100% | Compare GINE |

## Files Modified

### Core Implementation
- `src/data.py`: Added label-as-feature support and 5D edge attributes
- `src/train.py`: Added masking functions and progressive schedule
- `src/models/gat.py`: Added `edge_dim` parameter
- `src/models/gine.py`: Added `edge_dim` parameter
- `src/models/transformer.py`: Added `edge_dim` parameter

### Configuration Files (New)
- `configs/transformer_solver.yaml`
- `configs/gat_solver.yaml`
- `configs/gine_solver.yaml`
- `configs/baseline_no_masking.yaml`

## Expected Training Behavior

### Epochs 0-20 (Warmup)
- Masking rate: 0%
- All edge labels visible
- Model learns basic patterns and valid configurations

### Epochs 20-100 (Progressive Masking)
- Masking rate: 0% → 60%
- Gradually increasing difficulty
- Model learns constraint propagation

### Epochs 100-200 (Full Puzzle Solving)
- Masking rate: 60% → 100%
- Near-complete to complete inference
- Model must solve puzzles from scratch

### Final Epoch Performance
- At 100% masking, validation accuracy = true puzzle-solving capability
- Model sees only island capacities and graph structure
- Must infer all bridge assignments via constraint reasoning

## MLflow Tracking

All experiments log to the same "Hashi Graph GNN" experiment with:
- Standard metrics: `train_loss`, `train_acc`, `val_loss`, `val_acc`
- **New metric**: `masking_rate` (tracks progression over epochs)
- Easy comparison across architectures and masking strategies

## Next Steps

1. **Train all configs** and compare in MLflow
2. **Analyze masking curves**: How does accuracy degrade with masking rate?
3. **Experiment with schedules**: Try linear vs cosine vs step schedules
4. **Partial masking**: Test `end_rate=0.5` or `0.7` for partial inference
5. **Architecture comparison**: Which model (GAT/GINE/Transformer) handles masking best?

## Technical Notes

- Masking is applied **per batch** with random selection
- Only original puzzle edges are masked (meta edges always visible)
- Edge dimension is automatically determined from config
- Processed files include `_lbl` suffix when using labels as features
- All changes maintain full backward compatibility






