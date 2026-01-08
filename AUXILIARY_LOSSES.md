# Auxiliary Loss Implementation

This document describes the implementation of two auxiliary losses for the Hashi puzzle solver, as outlined in `plans/enhancements.md`.

## Overview

The implementation adds **constraint-aware auxiliary losses** that explicitly encode Hashi puzzle rules to guide the model toward valid solutions. These losses work alongside the standard cross-entropy classification loss.

## Implemented Losses

### 1. Degree Violation Loss (Island Counting)

**Purpose**: Enforces that the sum of bridges connected to each island equals the island's capacity.

**Implementation**: `src/losses.py::compute_degree_violation_loss()`

**How it works**:
- For each node (island), computes the predicted degree as the sum of predicted bridge values on incident edges
- Uses soft bridge values: `E[bridges] = 0*P(0) + 1*P(1) + 2*P(2)`
- Loss = MSE(predicted_degree, target_capacity)
- Only counts original puzzle edges (excludes meta/conflict edges via masking)

**Why MSE**: Penalizes large violations more than small ones, encouraging the model to get close to the target even early in training.

**Mathematical Formula**:
```
predicted_degree[node_i] = Σ(expected_bridge_value[edge_j])  for edges incident to node_i
loss = MSE(predicted_degree, node_capacity)
```

### 2. Bridge Crossing Loss (Mutual Exclusion)

**Purpose**: Enforces that geometrically intersecting edges cannot both have bridges (they are mutually exclusive).

**Implementation**: `src/losses.py::compute_crossing_loss()`

**How it works**:
- Pre-computed crossing edge pairs are stored in `edge_conflicts` during data processing
- For each pair of crossing edges (e1, e2):
  - Compute probability that bridge exists: `P(bridge) = P(label=1) + P(label=2)`
  - Loss = `P(e1_has_bridge) * P(e2_has_bridge)`
- If both edges have high probability, the product (and gradient) explodes, forcing the model to "pick a winner"

**Mathematical Formula**:
```
P_active[edge] = P(label=1) + P(label=2)
loss_pair = P_active[edge1] * P_active[edge2]
total_loss = mean(loss_pair) for all crossing pairs
```

**Current Limitation**: Edge conflicts may not batch correctly with PyTorch Geometric's default batching mechanism. When batching issues are detected, crossing loss gracefully falls back to 0 (disabled). This primarily affects the degree loss, which works correctly in all cases.

### 3. Combined Loss Function

**Implementation**: `src/losses.py::compute_combined_loss()`

Combines all three losses with configurable weights:

```python
total_loss = λ_ce * loss_ce + 
             λ_degree * loss_degree + 
             λ_crossing * loss_crossing
```

## Configuration

Loss weights are configured in `configs/masking_experiment.yaml`:

```yaml
training:
  loss_weights:
    ce: 1.0           # Cross-entropy (standard classification)
    degree: 0.1       # Degree violation (island counting)
    crossing: 0.5     # Bridge crossing (mutual exclusion)
```

**Recommended starting values** (from `plans/enhancements.md`):
- `ce`: 1.0 (baseline)
- `degree`: 0.1 (avoid conflicting with CE early in training)
- `crossing`: 0.5 (critical constraint, higher weight)

## Integration

### Files Modified

1. **`src/losses.py`** (NEW): Contains all loss function implementations
2. **`src/train.py`**: 
   - Imports and uses combined loss function
   - Extracts node capacities and edge conflicts from data
   - Logs individual loss components to MLflow
3. **`src/data.py`**: 
   - Extracts edge conflict indices during data processing
   - Stores conflicts in `Data.edge_conflicts` attribute
   - Handles backward compatibility for existing processed data
4. **`configs/masking_experiment.yaml`**: Added `loss_weights` configuration

### Training Loop Changes

The `train_epoch()` function now:
1. Computes combined loss with auxiliary losses
2. Tracks and logs individual loss components (CE, degree, crossing)
3. Prints loss breakdown when auxiliary losses are enabled

Example output:
```
Epoch: 001, Train Loss: 0.8675, Train Acc: 0.5843, ...
  -> Loss Components: CE=0.7823, Degree=0.0531, Crossing=0.0321
```

## Usage

### Training with Auxiliary Losses

```bash
uv run -m src.train --config configs/masking_experiment.yaml
```

The config already includes:
- `loss_weights` enabled with recommended values
- Progressive masking for curriculum learning
- Transformer architecture with attention

### Disabling Auxiliary Losses

To disable auxiliary losses (fallback to standard CE only):

```yaml
training:
  loss_weights:
    ce: 1.0
    degree: 0.0      # Disabled
    crossing: 0.0    # Disabled
```

## Testing

A test suite was created and validated (`test_losses.py`, now removed):
- ✅ Degree loss computation
- ✅ Crossing loss computation
- ✅ Combined loss function
- ✅ Proper handling of meta edges (exclusion via masking)

All tests passed successfully.

## Expected Benefits

From `plans/enhancements.md`:

1. **Degree Loss**:
   - Forces model to "learn to count" bridges
   - Respects fundamental constraint: `sum(bridges) = island_capacity`
   - Should improve constraint satisfaction rate

2. **Crossing Loss**:
   - Prevents invalid solutions where bridges cross
   - Reduces post-processing/correction needed
   - Teaches spatial reasoning about grid layout

3. **Combined**:
   - Guides optimization toward valid solution space
   - Complements classification loss with domain knowledge
   - Should improve both accuracy and perfect puzzle rate

## Future Enhancements

From `plans/enhancements.md`, not yet implemented:

1. **Edge Conflict Batching**: Properly adjust edge conflict indices when batching graphs
2. **Connectivity Loss**: Ensure predicted graph is fully connected (spectral or component-based)
3. **Edge Symmetry Loss**: Enforce bidirectional consistency (currently model predicts both directions independently)
4. **Adaptive Loss Weighting**: Dynamically adjust λ weights based on training phase
5. **Per-Edge Loss Caching**: For adaptive curriculum learning (easy-first vs. hard-first masking)

## MLflow Tracking

The following metrics are now logged:
- `train_loss`: Total combined loss
- `train_ce_loss`: Cross-entropy component
- `train_degree_loss`: Degree violation component
- `train_crossing_loss`: Bridge crossing component
- Same for validation (though val uses CE only for now)

This allows analyzing the contribution of each loss over time and tuning weights accordingly.

## References

- Design rationale: `plans/enhancements.md` Section 3.E (Constraint-Aware Auxiliary Losses)
- Similar techniques: Hard negative mining, soft constraints in physics-informed neural networks
- Hashi rules: https://en.wikipedia.org/wiki/Hashiwokakero




