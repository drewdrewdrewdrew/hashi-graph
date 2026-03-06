# Requirements: BPTT Training for Hashi Diffusion Solver

**Defined:** 2026-03-06
**Core Value:** The model learns multi-step coordination by receiving gradient signal that flows across consecutive diffusion steps

## v1 Requirements

### Config

- [x] **CFG-01**: `bptt.enabled` flag in YAML and `TrainingConfig` (default false, fully backward-compatible)
- [x] **CFG-02**: `bptt.window` — number of consecutive steps to backprop through
- [x] **CFG-03**: `bptt.stride` — step size for sliding window across the step sequence
- [x] **CFG-04**: `bptt.loss_ema_decay` — EMA decay for smoothing window-averaged loss scalar

### Training

- [ ] **TRN-01**: When `bptt.enabled`, forward pass caches graph state (edge logits) at each step boundary
- [ ] **TRN-02**: Sliding window iterates over step sequence with configured window + stride
- [ ] **TRN-03**: Within each window, gradient flows through consecutive step transitions (no `no_grad` block)
- [ ] **TRN-04**: Gradient checkpointing applied within each window to bound activation memory
- [ ] **TRN-05**: Gradients accumulate across overlapping windows before optimizer step
- [ ] **TRN-06**: Loss is averaged across steps within each window; EMA applied to that scalar

### Compatibility

- [ ] **COMP-01**: When `bptt.enabled: false`, training loop is byte-for-byte equivalent to current behavior
- [ ] **COMP-02**: All existing config fields (`num_inference_steps_training`, `n_blocks`, etc.) remain valid

## v2 Requirements

- **TRN-07**: Gradient norm logging per-window for diagnosing vanishing/exploding gradients across steps
- **TRN-08**: Warmup schedule for BPTT — start with small window, grow over epochs

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full-sequence BPTT (window = all steps) | Memory impractical; sliding window achieves the goal |
| EMA of model weights | User specified loss-value EMA only |
| Changes to eval/inference rollout | BPTT is training-only |
| New loss terms | Existing per-step losses are sufficient; BPTT changes gradient flow, not loss definition |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CFG-01 | Phase 1 | Complete |
| CFG-02 | Phase 1 | Complete |
| CFG-03 | Phase 1 | Complete |
| CFG-04 | Phase 1 | Complete |
| TRN-01 | Phase 2 | Pending |
| TRN-02 | Phase 2 | Pending |
| TRN-03 | Phase 2 | Pending |
| TRN-04 | Phase 2 | Pending |
| TRN-05 | Phase 2 | Pending |
| TRN-06 | Phase 2 | Pending |
| COMP-01 | Phase 2 | Pending |
| COMP-02 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-06*
*Last updated: 2026-03-06 after roadmap creation*
