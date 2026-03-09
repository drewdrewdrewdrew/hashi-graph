# Requirements: Hashi Diffusion Solver — Training Infrastructure

**Defined:** 2026-03-06
**Core Value:** The model learns to make decisions that are good for a sequence of steps, not just the next step — relaxing myopic per-step optimization with longer-horizon gradient signal and iterative constraint reasoning

## v1.0 Requirements (Complete)

### Config

- [x] **CFG-01**: `bptt.enabled` flag in YAML and `TrainingConfig` (default false, fully backward-compatible)
- [x] **CFG-02**: `bptt.window` — number of consecutive steps to backprop through
- [x] **CFG-03**: `bptt.stride` — step size for sliding window across the step sequence
- [x] **CFG-04**: `bptt.loss_ema_decay` — EMA decay for smoothing window-averaged loss scalar

### Training

- [x] **TRN-01**: When `bptt.enabled`, forward pass caches graph state (edge logits) at each step boundary
- [x] **TRN-02**: Sliding window iterates over step sequence with configured window + stride
- [x] **TRN-03**: Within each window, gradient flows through consecutive step transitions (no `no_grad` block)
- [x] **TRN-04**: Gradient checkpointing applied within each window to bound activation memory
- [x] **TRN-05**: Gradients accumulate across overlapping windows before optimizer step
- [x] **TRN-06**: Loss is averaged across steps within each window; EMA applied to that scalar

### Compatibility

- [x] **COMP-01**: When `bptt.enabled: false`, training loop is byte-for-byte equivalent to current behavior
- [x] **COMP-02**: All existing config fields (`num_inference_steps_training`, `n_blocks`, etc.) remain valid

## v1.1 Requirements

### Bug Fix

- [x] **BUG-01**: BPTT can be enabled alongside any training mode without a `scales` UnboundLocalError crash

### Config Schema

- [x] **CFG-05**: `ReasoningConfig` dataclass (`enabled: bool = False`, `steps: int = 5`) in `config.py` with `steps >= 1` validation
- [x] **CFG-06**: `ReverseGnnConfig` dataclass (`enabled: bool = False`, `separate_weights: bool = True`, `project_embeddings: bool = True`) in `config.py`
- [x] **CFG-07**: `ModelConfig` has typed `reasoning` and `reverse_gnn` fields, both defaulting to disabled
- [x] **CFG-08**: New `rev_reasoning.yaml` copied from `diffusion_solver_continuous_bptt.yaml` — diffusion params commented out, `num_inference_steps_training` dropped, `model.reasoning` and `model.reverse_gnn` blocks added, `training.mode: rev-reason`

### Training Mode

- [ ] **MODE-01**: `training.mode = "rev-reason"` routes to the new training path in `DiffusionTrainer.run_epoch` (no noise injection on edges)
- [ ] **MODE-02**: Within `rev-reason`, `reasoning.enabled` and `reverse_gnn.enabled` independently activate their components (either, both, or neither)

### Reasoning Component

- [x] **REAS-01**: When `reasoning.enabled: true`, a single shared-weight TransformerConv layer is applied K times with residual updates before the EdgeHead
- [x] **REAS-02**: Number of iterations controlled by `reasoning.steps`

### Reverse GNN Component

- [x] **REVG-01**: When `reverse_gnn.enabled: true`, a reverse backbone runs on the same input and its output is concatenated with forward embeddings
- [x] **REVG-02**: When `reverse_gnn.separate_weights: true`, reverse backbone has independent parameters from the forward backbone
- [x] **REVG-03**: When `reverse_gnn.project_embeddings: true`, a linear layer compresses concatenated embeddings back to `hidden_channels` before the EdgeHead

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
| Noise injection in rev-reason mode | rev-reason is pure graph classification; noise path belongs to diff-cont only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CFG-01 | Phase 1 | Complete |
| CFG-02 | Phase 1 | Complete |
| CFG-03 | Phase 1 | Complete |
| CFG-04 | Phase 1 | Complete |
| TRN-01 | Phase 2 | Complete |
| TRN-02 | Phase 2 | Complete |
| TRN-03 | Phase 2 | Complete |
| TRN-04 | Phase 2 | Complete |
| TRN-05 | Phase 2 | Complete |
| TRN-06 | Phase 2 | Complete |
| COMP-01 | Phase 2 | Complete |
| COMP-02 | Phase 2 | Complete |
| BUG-01 | Phase 3 | Complete |
| CFG-05 | Phase 3 | Complete |
| CFG-06 | Phase 3 | Complete |
| CFG-07 | Phase 3 | Complete |
| CFG-08 | Phase 3 | Complete |
| MODE-01 | Phase 4 | Pending |
| MODE-02 | Phase 4 | Pending |
| REAS-01 | Phase 4 | Complete |
| REAS-02 | Phase 4 | Complete |
| REVG-01 | Phase 4 | Complete |
| REVG-02 | Phase 4 | Complete |
| REVG-03 | Phase 4 | Complete |

**Coverage:**
- v1.0 requirements: 12 total — mapped: 12, unmapped: 0 ✓
- v1.1 requirements: 11 total — mapped: 11, unmapped: 0 ✓
- Phase 5 carries no new REQ-IDs; its success criteria validate end-to-end composability of Phase 4 components ✓

---
*Requirements defined: 2026-03-06*
*Last updated: 2026-03-09 after v1.1 roadmap revision (parallelized Phase 4)*
