# Roadmap: BPTT Training for Hashi Diffusion Solver

## Milestones

- ✅ **v1.0 BPTT** - Phases 1-2 (shipped 2026-03-06)
- 🚧 **v1.1 Reasoning** - Phases 3-5 (in progress)

## Phases

<details>
<summary>✅ v1.0 BPTT (Phases 1-2) - SHIPPED 2026-03-06</summary>

### Phase 1: Config Schema
**Goal**: The `bptt:` sub-block exists in YAML and is fully typed in `TrainingConfig`, so the training loop can read all BPTT parameters
**Depends on**: Nothing (first phase)
**Requirements**: CFG-01, CFG-02, CFG-03, CFG-04
**Success Criteria** (what must be TRUE):
  1. `diffusion_solver_continuous.yaml` contains a `bptt:` block with `enabled`, `window`, `stride`, and `loss_ema_decay` fields
  2. `TrainingConfig` dataclass has a typed `bptt` field (nested dataclass) with all four parameters and correct defaults
  3. Setting `bptt.enabled: false` (the default) causes no visible change to trainer behavior — existing training runs load without error
  4. Config validation rejects malformed values (e.g., window < 1, stride < 1)
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md — Add BpttConfig dataclass and bptt: YAML block

### Phase 2: BPTT Training Loop
**Goal**: When `bptt.enabled: true`, the diffusion training loop uses sliding-window backpropagation through time, with gradient checkpointing and window-loss EMA, while remaining byte-for-byte equivalent to current behavior when disabled
**Depends on**: Phase 1
**Requirements**: TRN-01, TRN-02, TRN-03, TRN-04, TRN-05, TRN-06, COMP-01, COMP-02
**Success Criteria** (what must be TRUE):
  1. With `bptt.enabled: false`, a training run produces identical loss values and optimizer steps to the unmodified `DiffusionTrainer`
  2. With `bptt.enabled: true`, the cached edge-logit state at each step boundary is used as the start point for window re-runs, and gradients are present on model parameters after `.backward()` calls spanning multiple steps
  3. Peak GPU memory during a BPTT-enabled run is bounded by gradient checkpointing — activation memory does not grow linearly with window size
  4. Overlapping windows each call `.backward()` before the optimizer steps, so steps covered by multiple windows accumulate gradient from all covering windows
  5. The loss scalar reported per training iteration is the EMA-smoothed window-averaged step loss
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md — BPTT dispatch + step-state cache + backward-compat guard (TRN-01, COMP-01, COMP-02)
- [x] 02-02-PLAN.md — Sliding-window loop with gradient checkpointing, accumulation, and EMA (TRN-02 through TRN-06)

</details>

### 🚧 v1.1 Reasoning (In Progress)

**Milestone Goal:** Add reasoning (iterative shared-weight message passing) and reverse GNN (parallel reverse backbone) as independently toggleable training modes, composable as `rev-reasoning`, reusing existing trainer and backbone infrastructure.

- [x] **Phase 3: Config Schema + Bug Fix** - Add `ReasoningConfig`/`ReverseGnnConfig` dataclasses, `rev_reasoning.yaml`, and fix the `scales` UnboundLocalError. Serial prerequisite for all implementation work. (completed 2026-03-09)
- [ ] **Phase 4: Component Implementation** - Three parallel plans implementing trainer dispatch and both new model components (no shared file writes)
- [ ] **Phase 5: Integration** - Wire `IterativeBackbone` and `ReverseBackbone` into `HashiGraphModel.forward()` and adapt `EdgeHead` for variable input dimensions

## Phase Details

### Phase 3: Config Schema + Bug Fix
**Goal**: Config types for both new components exist in `config.py`, `ModelConfig` has typed fields for both, a reference YAML is in place, and the `scales` crash is eliminated — so no implementation code can reference `reasoning` or `reverse_gnn` without a typed home
**Depends on**: Phase 2
**Requirements**: BUG-01, CFG-05, CFG-06, CFG-07, CFG-08
**Success Criteria** (what must be TRUE):
  1. `ReasoningConfig` and `ReverseGnnConfig` dataclasses exist in `config.py` with correct fields and defaults; all existing configs load without error
  2. `ModelConfig` has typed `reasoning: ReasoningConfig` and `reverse_gnn: ReverseGnnConfig` fields, both defaulting to disabled
  3. `rev_reasoning.yaml` exists with `training.mode: rev-reason` and both config blocks; no noise/diffusion params
  4. Running any training mode alongside `bptt.enabled: true` no longer crashes with `UnboundLocalError: scales`
**Plans**: 1 plan

Plans:
- [ ] 03-01-PLAN.md — Add ReasoningConfig, ReverseGnnConfig, update ModelConfig, fix BUG-01, create rev_reasoning.yaml

### Phase 4: Component Implementation
**Goal**: The trainer's `rev-reason` dispatch path exists and both new model components (`IterativeBackbone`, `ReverseBackbone`) exist as standalone classes — each independently verifiable, no integration into `HashiGraphModel.forward()` yet
**Depends on**: Phase 3
**Requirements**: MODE-01, MODE-02, REAS-01, REAS-02, REVG-01, REVG-02, REVG-03
**Parallelization**: Plans 04-01, 04-02, 04-03 touch different files and have no shared write conflicts — execute in parallel
**Success Criteria** (what must be TRUE):
  1. (04-01) `rev-reason` routes to a trainer path with no noise injection on edges
  2. (04-01) Each component flag (`reasoning.enabled`, `reverse_gnn.enabled`) routes independently; neither enabled runs a plain forward pass without error
  3. (04-02) `IterativeBackbone` class exists: single shared-weight TransformerConv, residual update, configurable K iterations
  4. (04-02) Instantiating `IterativeBackbone` with `steps=1` produces output identical to a single non-iterative forward pass
  5. (04-03) `ReverseBackbone` class exists: accepts same inputs as forward backbone, returns embeddings of same shape
  6. (04-03) `separate_weights=True` gives independent parameters; `separate_weights=False` shares forward backbone weights
  7. (04-03) `project_embeddings=True` adds a linear layer that compresses output to `hidden_channels`
**Plans**: 3 plans (parallel)

Plans:
- [ ] 04-01-PLAN.md — Rev-reason dispatch in `trainers/diffusion.py` (MODE-01, MODE-02) — file: `trainers/diffusion.py`
- [ ] 04-02-PLAN.md — `IterativeBackbone` class (REAS-01, REAS-02) — file: new class file (no changes to existing classes)
- [ ] 04-03-PLAN.md — `ReverseBackbone` class (REVG-01, REVG-02, REVG-03) — file: new class file (no changes to existing classes)

### Phase 5: Integration
**Goal**: `HashiGraphModel.forward()` composes `IterativeBackbone` and `ReverseBackbone` based on config flags, `EdgeHead` handles variable input dimensions in all flag combinations, and the full system is end-to-end verifiable
**Depends on**: Phase 4 (all 3 plans complete)
**Requirements**: (none new — delivers composability that makes Phase 4 requirements end-to-end verifiable)
**Success Criteria** (what must be TRUE):
  1. `HashiGraphModel.forward()` composes `IterativeBackbone` and `ReverseBackbone`: reasoning runs K iterations, reverse output is concatenated, projection applied if enabled
  2. With all flags disabled, `HashiGraphModel.forward()` produces byte-for-byte identical output to the pre-phase baseline
  3. With both enabled (rev-reasoning), each reasoning iteration uses forward + reverse passes before the residual update
  4. `EdgeHead` receives the correct input dimension in all flag combinations (no shape mismatch errors)
**Plans**: 1 plan

Plans:
- [ ] 05-01-PLAN.md — Wire IterativeBackbone and ReverseBackbone into HashiGraphModel.forward(), update EdgeHead for variable input dimensions

## Progress

**Execution Order:**
Phases 3 and 5 are serial. Phase 4 plans execute in parallel (no file conflicts).

```
Phase 3 (serial)
    └── Phase 4: 04-01 ┐
                  04-02 ├── (parallel)
                  04-03 ┘
                    └── Phase 5 (serial)
```

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Config Schema | v1.0 | 1/1 | Complete | 2026-03-06 |
| 2. BPTT Training Loop | v1.0 | 2/2 | Complete | 2026-03-06 |
| 3. Config Schema + Bug Fix | 1/1 | Complete   | 2026-03-09 | - |
| 4. Component Implementation | v1.1 | 0/3 | Not started | - |
| 5. Integration | v1.1 | 0/1 | Not started | - |
