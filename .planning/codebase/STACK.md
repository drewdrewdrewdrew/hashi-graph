# Technology Stack

**Analysis Date:** 2026-03-06

## Languages

**Primary:**
- Python 3.11 - All source code, training scripts, data pipeline, model definitions

**Secondary:**
- C - Native puzzle generator (`generator/bridges_gen.c`, `generator/bridges.c`, etc.) — compiled to a binary invoked via subprocess from `src/hashi_puzzle_solver/bridges_gen.py`

## Runtime

**Environment:**
- Python 3.11 (pinned via `.python-version`)

**Package Manager:**
- `uv` — used for virtual environment creation and dependency sync
- Lockfile: `uv.lock` (present and committed)
- `setup_env.sh` handles platform-specific torch/CUDA installation via `uv pip install` with custom index URLs

## Frameworks

**Core ML:**
- PyTorch 2.9.1 — tensor operations, model definition, training loop
- PyTorch Geometric (torch-geometric) 2.7.0 — graph neural network layers (`TransformerConv`, `global_mean_pool`), `Data`/`Dataset`/`DataLoader`/`Batch` abstractions
- SciPy 1.16.3 — sparse linear algebra for spectral graph features (`scipy.sparse.linalg`)

**GNN Model Architectures (in `src/hashi_puzzle_solver/models/`):**
- `transformer.py` — `TransformerEdgeClassifier` using `TransformerConv`
- `gine.py` — `GINEEdgeClassifier` using GINE convolution
- `gcn.py` — GCN-based classifier
- `gat.py` — GAT-based classifier
- `factory.py` — `ModelFactory` to instantiate any architecture from config

**Hyperparameter Tuning:**
- Optuna 4.6.0 — trial-based search in `src/hashi_puzzle_solver/tune.py`

**Experiment Tracking:**
- MLflow 3.7.0 — local SQLite-backed tracking (`mlflow.db`) with experiment/run logging via `MLflowCallback` in `src/hashi_puzzle_solver/callbacks.py`

**Data / Numerics:**
- NetworkX 3.6 — graph construction and analysis (centrality, articulation points)
- NumPy 2.3.5 — feature arrays, spectral computations
- Pandas 2.3.3 — tabular analysis in evaluation/scripts

**Visualization:**
- Matplotlib 3.10.7 — plots in notebooks and scripts

**Progress:**
- tqdm 4.67.1 — dataset processing progress bars

**Development / Notebooks:**
- JupyterLab 4.3.6 / Jupyter 1.1.1 — exploratory notebooks in `notebooks/`
- ipykernel 6.29.5

**Testing:**
- pytest 9.0.2 — test runner; config in `pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["src"]`)

**Build/Linting:**
- Ruff 0.14.10 — formatting and linting (configured in `pyproject.toml` with near-comprehensive rule set, target `py311`, line length 88, numpy docstring convention)

## Key Dependencies

**Critical:**
- `torch` 2.9.1 — entire model definition and training depends on it
- `torch-geometric` 2.7.0 — all graph data structures and GNN layers
- `hashi` 0.1.0 — third-party Hashiwokakero puzzle library (`generate`, `print_solution`), used as a secondary generator alongside the C binary
- `mlflow` 3.7.0 — experiment persistence; all training runs log to local `mlflow.db` SQLite file
- `optuna` 4.6.0 — hyperparameter sweep driver in `tune.py`

**Infrastructure:**
- `python-dotenv` 1.0.1 — `.env` file loading (present in dependencies; `.env` file existence not confirmed in repo root)
- `PyYAML` 6.0.2 — all config loading (`configs/*.yaml` parsed via `yaml.safe_load`)
- `scipy` 1.16.3 — spectral fingerprinting features in `data.py`
- `networkx` 3.6 — graph feature extraction (closeness centrality, articulation points)

## Configuration

**Environment:**
- All training/model/data settings are in YAML config files under `configs/`
- Key configs: `configs/base_config.yaml`, `configs/diffusion_solver_continuous.yaml`, `configs/diffusion_solver_discrete.yaml`, `configs/ar_solver.yaml`, `configs/transformer_solver.yaml`, `configs/gat_solver.yaml`, `configs/gine_solver.yaml`, `configs/tune_config.yaml`
- Training mode controlled by `training.mode`: one of `"one-shot"`, `"ar"`, `"diff-discrete"`, `"diff-cont"`
- Device selection: `training.device`: `"auto"`, `"cpu"`, `"cuda"`, or `"mps"`
- `python-dotenv` is a dependency, but no `.env` file is present in the repo root (secrets/env vars are not required for local training)

**Build:**
- `pyproject.toml` — project metadata, dependency declarations, pytest config, ruff config
- `setup_env.sh` — environment bootstrap script; detects CUDA/macOS/CPU and installs correct torch wheel

## Platform Requirements

**Development:**
- Python 3.11
- `uv` package manager
- Optional: NVIDIA GPU (CUDA) or Apple Silicon (MPS) for accelerated training
- C compiler (gcc/make) required only if rebuilding the `generator/bridges_gen` C binary

**Production:**
- No web server or deployment target detected — this is a standalone research/training codebase
- Models are saved as `.pt` files under `models/model_<timestamp>/`
- MLflow tracking is local-only (SQLite `mlflow.db` at project root, artifact dir `.mlop/`)

---

*Stack analysis: 2026-03-06*
