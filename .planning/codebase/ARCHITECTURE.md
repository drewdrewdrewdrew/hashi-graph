# Architecture

**Analysis Date:** 2026-03-06

## Pattern Overview

**Overall:** Layered ML pipeline with strategy-based training modes

**Key Characteristics:**
- GNN-based edge classification over puzzle graphs (predict bridge count 0/1/2 per edge)
- Three interchangeable training strategies (one-shot, auto-regressive, diffusion) dispatched via a Facade `Trainer`
- Component-assembled neural network: feature managers → encoders → backbone → heads, all wired by a `ModelFactory`
- Config-driven: a single YAML drives all model, training, and data decisions via typed dataclasses (`HashiModelConfig`)
- Active refactor: canonical code now lives in `src2/`; `src/` is legacy

## Layers

**Puzzle Generation Layer:**
- Purpose: Generate raw Hashi puzzle data from a C binary wrapper
- Location: `generator/` (C sources), `src2/hashi_puzzle_solver/bridges_gen.py`
- Contains: C binary `generator/bridges_gen`, Python subprocess wrapper `BridgesPuzzle` dataclass
- Depends on: OS subprocess call to compiled binary
- Used by: `src2/hashi_puzzle_solver/create_data.py`

**Data Preparation Layer:**
- Purpose: Convert raw JSON puzzles to PyG graph objects on disk; load them for training
- Location: `src2/hashi_puzzle_solver/create_data.py`, `src2/hashi_puzzle_solver/data.py`
- Contains: `HashiDataset` (PyG `Dataset` subclass), `HashiDatasetCache`, `FeatureSchema`, `RandomHashiAugment`
- Depends on: `generator/bridges_gen` (for generation), `utils/graph_utils.py` and `utils/bridges_utils.py` for graph construction
- Used by: `BaseTrainer.create_dataloader()`

**Configuration Layer:**
- Purpose: Typed, validated configuration shared across all components
- Location: `src2/hashi_puzzle_solver/models/config.py`
- Contains: `HashiModelConfig`, `DataConfig`, `ModelConfig`, `TrainingConfig`, `LossWeightsConfig`, `EarlyStoppingConfig`
- Depends on: Python stdlib `dataclasses`, `yaml`
- Used by: All layers; loaded from YAML via `utils/common.py:load_config()`

**Model Layer:**
- Purpose: Compose the GNN from pluggable sub-components
- Location: `src2/hashi_puzzle_solver/models/`
- Contains: `HashiGraphModel` (shell), `NodeEncoder`, `EdgeEncoder`, `GraphBackbone`, `EdgeHead`, `ProphetHead`, `NodeFeatureManager`, `EdgeFeatureManager`, `ModelFactory`
- Depends on: `torch_geometric`, `config.py`
- Used by: All trainers via `ModelFactory.create_model()`

**Loss Layer:**
- Purpose: Compute weighted combination of task and auxiliary losses
- Location: `src2/hashi_puzzle_solver/losses/`
- Contains: `HashiLossCalculator`, `DegreeLoss`, `CrossingLoss`, `VerificationLoss`, `LossModule` base, `legacy.py`
- Depends on: `torch`, model config for weights
- Used by: All trainer `run_epoch()` implementations

**Training Layer:**
- Purpose: Execute training loop for a chosen strategy
- Location: `src2/hashi_puzzle_solver/trainers/`, `src2/hashi_puzzle_solver/engine.py`
- Contains: `BaseTrainer`, `OneShotTrainer`, `ARTrainer`, `DiffusionTrainer`, `Trainer` facade, `EarlyStopper`, `EpochMetrics`
- Depends on: Model layer, Loss layer, Data layer, `masking.py`
- Used by: `src2/hashi_puzzle_solver/train.py` (entry point)

**Callback/Observability Layer:**
- Purpose: Side-effects during training (checkpointing, MLflow logging, console output)
- Location: `src2/hashi_puzzle_solver/callbacks.py`
- Contains: `CheckpointCallback`, `MLflowCallback`, `PrintMetricsCallback`, `OptunaPruningCallback`
- Depends on: `mlflow`, `torch`
- Used by: `train.py` assembles callbacks and passes them to `Trainer`

**Utilities Layer:**
- Purpose: Shared helper functions used across layers
- Location: `src2/hashi_puzzle_solver/utils/`
- Contains: `common.py` (device, config loading, collation), `train_utils.py`, `ar_utils.py`, `diffusion_utils.py`, `evaluation_utils.py`, `graph_utils.py`, `bridges_utils.py`
- Depends on: `torch`, `networkx`, `numpy`
- Used by: trainers, dataset, model components

## Data Flow

**Training Run:**

1. `src2/hashi_puzzle_solver/train.py:main()` parses CLI args and loads YAML config via `utils/common.py:load_config()`
2. `HashiModelConfig.from_dict(config)` produces typed config; device resolved via `get_device()`
3. Callbacks (`MLflowCallback`, `CheckpointCallback`, `PrintMetricsCallback`) instantiated
4. `Trainer(config, device, callbacks)` calls `create_trainer()` factory, which dispatches to `OneShotTrainer`, `ARTrainer`, or `DiffusionTrainer` based on `config.training.mode`
5. `trainer.train(train_transform)` calls `BaseTrainer._setup()`: creates model via `ModelFactory.create_model()`, optimizer (Adam), and two `DataLoader`s via `create_dataloader()`
6. Per epoch: `run_epoch(train_loader, training=True)` → batches flow through `MaskingStrategy.apply()` → model forward pass → `HashiLossCalculator()` → `loss.backward()` → optimizer step
7. Validation: `run_epoch(val_loader, training=False)` + optional rollout for AR/diffusion modes
8. Callbacks fire at `on_epoch_end()` with `EpochMetrics`; `CheckpointCallback` saves `model_latest.pt`; best model saved to `model_best.pt` on improvement

**Dataset Construction (offline):**

1. `src2/hashi_puzzle_solver/create_data.py` calls `generate_bridges()` (Python wrapper → `generator/bridges_gen` C binary via subprocess)
2. Each puzzle serialized as JSON to `dataset/raw/puzzle_NNNNN.json` with split label
3. At training time, `HashiDataset` loads raw JSON, calls `generate_puzzle_graph()` to build a NetworkX graph, converts to PyG `Data` objects with feature tensors, caches processed graphs to `dataset/processed_*/`

**Model Forward Pass:**

1. `NodeEncoder` projects raw node features (capacity, degree, structural, spectral) → `hidden_channels` embedding
2. `EdgeEncoder` projects edge features (distance, conflict, edge type, labels) → `edge_attr_dim` embedding
3. Optional: noise vector projected to `noise_embedding_dim`, injected into global meta node and/or edge attributes
4. `GraphBackbone` runs N-layer message passing (TransformerConv / GATConv / GINEConv) with residual connections
5. `EdgeHead` concatenates source/dest node embeddings + optional global meta + edge features → 3-class logits (0, 1, or 2 bridges)
6. Optional: `ProphetHead` predicts noise signal level for diffusion training objective

**State Management:**
- No global state; all state held in `BaseTrainer` instance fields (`model`, `optimizer`, `train_loader`, `val_loader`, `current_masking_rate`, `best_monitored_value`)
- AR mode uses `ARState` object per puzzle to track incremental bridge assignments during rollout
- Diffusion mode uses `carry_over_buffer_train` / `carry_over_buffer_val` lists in `DiffusionTrainer` for recursive batching

## Key Abstractions

**HashiGraphModel:**
- Purpose: Neural network shell that owns encoders, backbone, and heads
- Examples: `src2/hashi_puzzle_solver/models/core.py`
- Pattern: Composition — all sub-modules injected at construction by `ModelFactory`

**BaseTrainer:**
- Purpose: Shared setup, dataloader creation, epoch loop, early stopping, model save/load
- Examples: `src2/hashi_puzzle_solver/trainers/base.py`
- Pattern: Template Method — `run_epoch()` left abstract for subclasses

**HashiLossCalculator:**
- Purpose: Orchestrate CE + auxiliary losses (degree, crossing, verification) with configured weights
- Examples: `src2/hashi_puzzle_solver/losses/calculator.py`
- Pattern: Callable object; returns `dict[str, Tensor]` keyed by loss name

**ModelFactory:**
- Purpose: Centralized construction of `HashiGraphModel` from config
- Examples: `src2/hashi_puzzle_solver/models/factory.py`
- Pattern: Static factory method; builds feature managers → encoders → backbone → heads → model

**HashiModelConfig:**
- Purpose: Single root config object parsed from YAML; used by every layer
- Examples: `src2/hashi_puzzle_solver/models/config.py`
- Pattern: Nested dataclasses with `from_dict()` and `from_yaml()` class methods

**FeatureSchema:**
- Purpose: Named index map for node/edge feature tensors; avoids hardcoded column indices
- Examples: `src2/hashi_puzzle_solver/data.py`
- Pattern: Lookup object with `get_node_idx(name)` / `get_edge_idx(name)` methods

**MaskingStrategy:**
- Purpose: Curriculum learning via progressive edge masking schedules (cosine, linear, etc.)
- Examples: `src2/hashi_puzzle_solver/masking.py`
- Pattern: Strategy object; `get_rate(epoch, total_epochs)` + `apply(data, rate, device)`

## Entry Points

**Training:**
- Location: `src2/hashi_puzzle_solver/train.py:main()`
- Triggers: `python -m hashi_puzzle_solver.train --config configs/base_config.yaml`
- Responsibilities: Parse args, load config, set device/threading, assemble callbacks, create `Trainer`, call `train()`

**Hyperparameter Tuning:**
- Location: `src2/hashi_puzzle_solver/tune.py:main()`
- Triggers: `python -m hashi_puzzle_solver.tune --config configs/tune_config.yaml`
- Responsibilities: Launch Optuna study, run trials using `expand_trial_config()` from `tune_space.py`, log to MLflow

**Dataset Creation:**
- Location: `src2/hashi_puzzle_solver/create_data.py`
- Triggers: `python -m hashi_puzzle_solver.create_data` (CLI)
- Responsibilities: Generate puzzles via `bridges_gen` wrapper, assign splits, write JSON to `dataset/raw/`

**Evaluation:**
- Location: `scripts/evaluate_model.py`
- Triggers: CLI script
- Responsibilities: Load trained model, run inference on dataset with noise presets, produce diagnostic metrics

**Rollout Analysis:**
- Location: `scripts/tune_rollout.py`, `scripts/sweep_merge_margin.py`, `scripts/find_and_render_puzzles.py`
- Triggers: CLI scripts
- Responsibilities: Post-training analysis and hyperparameter sweeps

## Error Handling

**Strategy:** Minimal; exceptions propagate naturally. Training loop uses `try/finally` to ensure `on_train_end` callbacks always fire even on crash.

**Patterns:**
- `FeatureSchema.get_node_idx()` / `get_edge_idx()` raise `ValueError` with available keys listed
- `create_trainer()` silently falls through to `OneShotTrainer` for unrecognized modes
- Device selection in `get_device()` falls back through cuda → mps → cpu

## Cross-Cutting Concerns

**Logging:** MLflow via `MLflowCallback`; console via `PrintMetricsCallback` and `tqdm` progress bars
**Validation:** Config parsed to typed dataclasses at startup (fails fast on unknown keys); feature schema enforces named access
**Device Handling:** `utils/common.py:get_device()` resolves `"auto"` to best available device; `pin_memory` and `torch.compile` conditionally applied based on device type

---

*Architecture analysis: 2026-03-06*
