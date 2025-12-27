# Critical Bugfixes - Edge Masking & Conflict Batching

## Bug #1: Edge Label Masking Indices (CRITICAL)

### Problem
Hardcoded indices for masking bridge labels broke when new edge features were added.

**Before (working with 5 features):**
```python
data.edge_attr[mask_indices, 3] = 0.0  # bridge_label
data.edge_attr[mask_indices, 4] = 0.0  # is_labeled
```

**After adding use_meta_mesh/use_meta_row_col_edges (7 features):**
```
Index: 0     1     2        3           4            5                   6            7
      [dx,   dy,   meta,    conflict,   meta_mesh,   meta_row_col_cross, bridge_lbl,  is_labeled]
                            ↑           ↑
                            Wrong!      Wrong!
```

Indices 3-4 were masking **conflict flags** instead of **bridge labels**!

### Impact
- ❌ Validation had **full access to ground truth** during masking
- ❌ Val accuracy artificially inflated (68% when should be ~33% random)
- ❌ Curriculum learning completely broken
- ❌ Model never learned true puzzle-solving

### Fix
Dynamic index calculation:
```python
edge_dim = data.edge_attr.size(1)
bridge_label_idx = edge_dim - 2  # Always second-to-last
is_labeled_idx = edge_dim - 1     # Always last
```

Now works regardless of which features are enabled!

---

## Bug #2: Edge Conflicts Not Batching

### Problem
PyTorch Geometric's `DataLoader` doesn't know how to batch custom list attributes like `edge_conflicts`.

**What happened:**
```python
# Graph 1: edge_conflicts = [(0, 5), (2, 7)]
# Graph 2: edge_conflicts = [(1, 4), (3, 6)]

# After default batching:
# - Graph 2's edges now start at index 100 (offset by Graph 1's edge count)
# - But edge_conflicts still has original indices: [(0,5), (2,7), (1,4), (3,6)]
# - Indices point to wrong edges! Or conflicts become None/corrupted
```

### Impact
- ❌ Crossing loss always **0.0** (no valid conflicts)
- ❌ Model not learning mutual exclusion constraint
- ❌ Can predict crossing bridges without penalty

### Fix
Custom collate function with proper index offsetting:

```python
def custom_collate_with_conflicts(data_list):
    batch = Batch.from_data_list(data_list)
    
    all_conflicts = []
    edge_offset = 0
    
    for data in data_list:
        if data.edge_conflicts:
            # Adjust indices by edge offset
            for e1, e2 in data.edge_conflicts:
                all_conflicts.append((e1 + edge_offset, e2 + edge_offset))
        
        edge_offset += data.edge_index.size(1)
    
    batch.edge_conflicts = all_conflicts
    return batch
```

---

## Expected Training Behavior After Fixes

### Validation (with 100% masking)
- **Epoch 1**: Val Acc ~33-40% (near random, model guessing)
- **Should gradually improve** as model learns reasoning
- **Perfect puzzle rate** should start near 0% and climb slowly

### Loss Components
- **CE Loss**: Standard classification (was working)
- **Degree Loss**: Island counting (was working: 0.34 → 0.09 → 0.07)
- **Crossing Loss**: NOW WORKING (should see non-zero values)

### Before vs After
```
BEFORE (Broken):
Epoch 1: Val Acc 68%, Val Perfect 0.88%, Crossing Loss 0.0

AFTER (Fixed):
Epoch 1: Val Acc ~35-40%, Val Perfect ~0.0%, Crossing Loss ~0.15
```

---

## Files Modified

1. **src/train.py**:
   - Fixed `apply_edge_label_masking()` with dynamic indices
   - Added `custom_collate_with_conflicts()` function
   - Updated DataLoaders to use custom collate

2. **src/losses.py**:
   - Simplified `compute_crossing_loss()` validation (collate handles it now)

---

## Testing Recommendations

After retraining with fixed code:

1. **Check validation starts low**: Epoch 1 val acc should be 33-40%, not 68%
2. **Check crossing loss activates**: Should see values like 0.05-0.20, not 0.0
3. **Monitor degree loss**: Should continue improving (was already working)
4. **Watch val accuracy climb**: Should gradually improve over epochs

If validation is still suspiciously high on epoch 1, there may be other label leakage paths to investigate.

---

## Prevention

Going forward, when adding new edge features:

1. **Always add at the END** (before bridge_label/is_labeled)
2. **Never use hardcoded indices** for dynamic structures
3. **Test with different feature combinations** to ensure robustness
4. **Custom attributes need custom collate** functions for batching




