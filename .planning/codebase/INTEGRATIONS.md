# External Integrations

**Analysis Date:** 2026-03-06

## APIs & External Services

**None detected.** The codebase makes no outbound HTTP calls to external APIs at runtime. All data is local (JSON puzzle files in `dataset/raw/`) and all computation is local. No cloud service SDKs are imported in the source.

## Data Storage

**Databases:**
- None (no relational database, ORM, or database client)

**Flat File Storage:**
- Puzzle dataset: `dataset/raw/puzzle_XXXXXXXX.json` — JSON files, one per puzzle, loaded by `src/hashi_puzzle_solver/data.py`
- Model checkpoints: `models/model_<timestamp>/model_latest.pt` and `models/model_<timestamp>/config.yaml` — written by `CheckpointCallback` in `src/hashi_puzzle_solver/callbacks.py`

**MLflow Tracking Store:**
- Backend: SQLite file `mlflow.db` at project root
- Artifact store: `.mlop/hashi-graph/` directory (local filesystem)
- Client: `mlflow` Python SDK — no remote tracking server configured; default local file-based tracking is used
- Configured via: `MLflowCallback` in `src/hashi_puzzle_solver/callbacks.py` — sets experiment name `"Hashi Graph GNN"` and logs params/metrics per epoch

**Caching:**
- In-memory singleton: `HashiDatasetCache` in `src/hashi_puzzle_solver/data.py` — caches processed `HashiDataset` objects in a class-level dict keyed by a hash of the data/model config and split name; no persistent disk cache beyond the raw JSON files

## Authentication & Identity

**Auth Provider:** None — no authentication layer anywhere in the codebase.

## Monitoring & Observability

**Experiment Tracking:**
- MLflow 3.7.0 — local tracking only; metrics logged per epoch include loss components (CE, degree, crossing, noise), edge accuracy, and perfect puzzle accuracy
- MLflow UI can be launched locally with `mlflow ui --backend-store-uri sqlite:///mlflow.db` (not scripted, but `mlflow.db` exists at project root)
- Optuna 4.6.0 — hyperparameter tuning trials with MLflow integration via `MLflowCallback` and `OptunaPruningCallback` (referenced in `src/hashi_puzzle_solver/tune.py`)

**Error Tracking:** None (no Sentry, Rollbar, etc.)

**Logs:**
- `PrintMetricsCallback` in `src/hashi_puzzle_solver/callbacks.py` prints a formatted epoch metrics table to stdout
- No structured logging framework; all logging is via `print()`

## CI/CD & Deployment

**Hosting:** Not applicable — no deployment target. This is a standalone ML research project run locally.

**CI Pipeline:** None detected (no `.github/`, `.gitlab-ci.yml`, or equivalent).

## Environment Configuration

**Required env vars:** None confirmed as required. `python-dotenv` is a dependency but no `.env` file is present in the repo root. All configuration is via YAML files under `configs/`.

**Secrets location:** Not applicable — no secrets required for local training.

## Puzzle Generation

**C Binary (primary generator):**
- Source: `generator/bridges_gen.c`, `generator/bridges.c`, etc. (C source with `Makefile`)
- Python wrapper: `src/hashi_puzzle_solver/bridges_gen.py` — invokes the compiled `generator/bridges_gen` binary via `subprocess`
- Output: puzzle param strings and solution strings

**`hashi` Python package (secondary generator):**
- Package: `hashi` 0.1.0 (PyPI)
- Used in: `src/hashi_puzzle_solver/yo2.py` via `from hashi import print_solution`
- Provides: `generate(width, height, difficulty)` and `print_solution()`

## Webhooks & Callbacks

**Incoming:** None.
**Outgoing:** None.

---

*Integration audit: 2026-03-06*
