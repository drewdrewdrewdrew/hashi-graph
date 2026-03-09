# Codebase Concerns

**Analysis Date:** 2026-03-06

---

## Tech Debt

**Parallel `src` / `src2` codebases:**
- Issue: Two complete, near-duplicate implementations of the same package (`src/hashi_puzzle_solver/` and `src2/hashi_puzzle_solver/`) exist side-by-side. `src2` is a refactored version with better module decomposition (losses/, trainers/, utils/ sub-packages), but `src` is still the active, tested, and installed package.
- Files: `src/hashi_puzzle_solver/` (all files), `src2/hashi_puzzle_solver/` (all files), `pyproject.toml` (only configures `src`), `tests_src2/conftest.py` (injects `src2` via `sys.path.insert` to shadow `src`)
- Impact: Every fix, feature, or experiment applied to `src` must be mirrored to `src2` (or vice versa) or the two drift. The `tests_src2/` suite tests `src2` but the production training entry point runs `src`. New contributors cannot tell which codebase is canonical.
- Fix approach: Designate one as canonical (likely `src2` given its cleaner structure), migrate remaining `src`-only functionality (notably `evaluate_model.py`, `ar_engine.py`, `diffusion_engine.py`), then delete `src` and the `src2` workaround in `tests_src2/conftest.py`.

**`evaluate_model.py` uses deprecated model classes:**
- Issue: `src/hashi_puzzle_solver/evaluate_model.py` imports and instantiates `GATEdgeClassifier` and `GCNEdgeClassifier`, which are no longer exported from `src/hashi_puzzle_solver/models/__init__.py` and are not supported by `ModelFactory`. It also recalculates `edge_dim` manually using a subset of flags, missing newer features such as `use_meta_mesh`, `use_meta_row_col_edges`, `use_continuous_edge_labels`, etc.
- Files: `src/hashi_puzzle_solver/evaluate_model.py` lines 22-125
- Impact: Loading any recent checkpoint with the standalone `evaluate_model.py` script will fail or silently produce wrong architecture if the model config uses modern feature flags.
- Fix approach: Replace the manual model construction in `evaluate_model.py` with `ModelFactory.create_model(config, device)`, matching what the `Trainer` does.

**`yo2.py` scratch file committed to `src`:**
- Issue: `src/hashi_puzzle_solver/yo2.py` is an early scratch/exploration file (14 lines, uses removed `src.bridges_utils` and `hashi` package, contains informal TODO notes in a docstring). It is part of the installed package.
- Files: `src/hashi_puzzle_solver/yo2.py`
- Impact: Confuses package consumers; imports will fail at runtime if executed.
- Fix approach: Delete the file.

**`limit` parameter semantics silently changed:**
- Issue: The `limit` parameter in `HashiDataset.__init__` and all callers was redefined mid-development. It previously capped which files were indexed; now it is passed as `None` and the caller instead uses a `RandomSampler` / `SubsetRandomSampler`. The comment `# REDEFINED: Always index all files for dynamic subsampling` appears in three places but the parameter is still accepted (and applied) in `_get_filtered_filenames` if non-`None`, creating a split behavior.
- Files: `src/hashi_puzzle_solver/engine.py` line 469, `src/hashi_puzzle_solver/data.py` lines 124, 534, `src2/hashi_puzzle_solver/data.py` line 124
- Impact: Callers that pass `limit` directly to `HashiDataset` (e.g., old configs or scripts) still cap the file index silently. The config key `limit` still exists in `diffusion_solver_continuous.yaml` with the comment "Legacy limit applied to both splits when train_limit/val_limit are absent", which means config intent and code behavior are misaligned.
- Fix approach: Remove `limit` from `HashiDataset.__init__`, move dataset sub-sampling entirely to the `DataLoader` sampler layer in `Trainer.create_dataloader`, and update configs.

**`HashiDatasetCache` key does not include `root_dir`:**
- Issue: `HashiDatasetCache._config_hash` hashes only `data.size`, `data.difficulty`, `data.limit`, and model feature flags. It does not include `root_dir`. If two different dataset directories are used in the same process (e.g., during tuning or multi-run experiments), the cache returns the wrong dataset.
- Files: `src/hashi_puzzle_solver/data.py` lines 67-102, `src2/hashi_puzzle_solver/data.py` lines 67-102
- Impact: Stale cache hits when root directory changes within one Python process.
- Fix approach: Include `data.root_dir` in the `relevant_config` dict used to compute the hash.

**Hardcoded Apple M3 Pro thread count:**
- Issue: `optimize_cpu_threading` in `src/hashi_puzzle_solver/train.py` hardcodes `num_cores = 11` with the comment `# M3 Pro has 11 cores`, and sets this unconditionally for all `arm64` Macs.
- Files: `src/hashi_puzzle_solver/train.py` lines 33-37, `src2/hashi_puzzle_solver/train.py` lines 33-37
- Impact: Sub-optimal thread allocation on M1 (8 cores), M2 (8-12 cores), M3 Max (16 cores), or any other ARM device. On CUDA machines running in CPU fallback mode, the function also applies Mac-specific logic (`platform.system() == "Darwin"` is checked).
- Fix approach: Use `os.cpu_count()` or `psutil.cpu_count(logical=False)` rather than a hardcoded constant.

---

## Known Bugs

**Division by zero when `n_blocks` is set and `num_inference_steps_training = 1`:**
- Symptoms: `ZeroDivisionError` during the first training batch in `diff-cont` mode if `n_blocks` is non-`None` and `num_inference_steps_training` is `1`.
- Files: `src/hashi_puzzle_solver/diffusion_engine.py` line 595
- Trigger: Set `n_blocks: 4` and `num_inference_steps_training: 1` in config (the subsample step fires when `train_step == 0` with `denom = 1 - 1 = 0`).
- Workaround: Keep `num_inference_steps_training >= 2` whenever `n_blocks` is set. The active config uses `num_inference_steps_training: 10`, so this is currently dormant.

**Validation `SubsetRandomSampler` always selects first `N` samples:**
- Symptoms: Validation metrics evaluated on the same first-N samples every epoch — no diversity across runs.
- Files: `src/hashi_puzzle_solver/engine.py` lines 526-529
- Trigger: Any config with a `val_limit` (e.g., `diffusion_solver_continuous.yaml` sets `val_limit: 500`). `indices = list(range(num_samples))` always produces indices 0 through N-1; while `SubsetRandomSampler` shuffles batch order, it cannot sample items outside those indices.
- Workaround: None. The effective validation set is frozen to the first 500 sorted files.

---

## Security Considerations

**`torch.load(weights_only=False)` for all processed `.pt` files:**
- Risk: Using `weights_only=False` allows arbitrary Python object deserialization via `pickle`. A malicious or corrupted `.pt` file in `dataset/processed_*/` would execute arbitrary code at dataset load time.
- Files: `src/hashi_puzzle_solver/data.py` line 680, `src2/hashi_puzzle_solver/data.py` line 680
- Current mitigation: Dataset files are generated locally by `create_data.py` and stored in `dataset/`. There is no external download mechanism.
- Recommendations: Migrate to saving/loading plain tensor dicts and reconstruct `Data` objects manually, enabling `weights_only=True`. At minimum, document the assumption that processed files are trusted.

**`subprocess.run` with user-controlled arguments in `bridges_gen.py`:**
- Risk: `generate_bridges()` constructs a command list and passes it to `subprocess.run`. If `size`, `difficulty`, or `count` parameters are ever sourced from untrusted input (e.g., a web API endpoint), they could be used for argument injection.
- Files: `src/hashi_puzzle_solver/bridges_gen.py` lines 161-175, `src2/hashi_puzzle_solver/bridges_gen.py` lines 161-175
- Current mitigation: Parameters are integers only; the command list form (not shell=True) prevents shell injection. Risk is low in current usage (CLI/research scripts only).
- Recommendations: Add type validation on `size`, `difficulty`, `count` at the function entry point.

**`spec.loader.exec_module` in `create_data.py`:**
- Risk: `src2/hashi_puzzle_solver/create_data.py` line 578 dynamically executes the `profile_puzzles.py` script via `spec.loader.exec_module`. If the path to `profile_puzzles.py` is ever influenced by user input or an untrusted environment, it could execute arbitrary code.
- Files: `src2/hashi_puzzle_solver/create_data.py` lines 572-579
- Current mitigation: The path is computed relative to `__file__`; not user-controllable in current usage.
- Recommendations: Replace with a direct function import rather than dynamic module execution.

---

## Performance Bottlenecks

**Dataset initialization scans all 75,000 raw JSON files on every process start:**
- Problem: `HashiDataset._get_filtered_filenames` opens and reads every `puzzle_*.json` to check `split`, `size`, and `difficulty` filters. With 75,000 files in `dataset/raw/`, this runs on every trainer start, every tune trial, and every worker process.
- Files: `src/hashi_puzzle_solver/data.py` lines 500-537
- Cause: No index file; split/size metadata is inside the JSON, not in the filename.
- Improvement path: Generate a lightweight CSV index at dataset creation time. `create_data.py` already has all this metadata when writing files; add an `index.csv` with columns `[filename, split, size, difficulty]` and use it in `_get_filtered_filenames`.

**Spectral feature computation is expensive and silently falls back to zeros:**
- Problem: `scipy.sparse.linalg.eigsh` with `maxiter=1000` is called once per puzzle during `process()`. On small puzzles where `g_potential.number_of_nodes() <= k + 1` (4 nodes or fewer), it silently falls back to `[0.0, 0.0, 0.0]`. Any exception also falls back silently, making it impossible to distinguish "valid zeros" from "failed computation".
- Files: `src/hashi_puzzle_solver/data.py` lines 765-789
- Cause: Broad `except Exception` catch swallows all eigsh convergence failures.
- Improvement path: Log a warning with the puzzle ID when falling back. Consider pre-computing spectral features in `create_data.py` and storing in JSON to avoid recomputation at process time.

**`HashiDataset.get()` applies `MakeBidirectional` on every single item access:**
- Problem: The bidirectional edge expansion (doubling `edge_index`, `edge_attr`, `y`, `edge_mask`, `edge_type`) is applied in `get()` at load time rather than at `process()` time. This means every worker re-processes every sample on every epoch.
- Files: `src/hashi_puzzle_solver/data.py` lines 675-695
- Cause: The `_oneway` suffix in processed filenames documents the design intention, but the transform is applied dynamically rather than persisted.
- Improvement path: Move `MakeBidirectional` into the `process()` step (remove the `_oneway` distinction), or cache the expanded tensors to disk.

---

## Fragile Areas

**Edge feature index computation is duplicated across four locations:**
- Files: `src/hashi_puzzle_solver/ar_utils.py` (`get_edge_feature_indices`, lines 9-62), `src/hashi_puzzle_solver/masking.py` (manual index computation, lines 74-85), `src/hashi_puzzle_solver/models/factory.py` (`compute_edge_dim`, lines 108-148), `src/hashi_puzzle_solver/data.py` (`_get_feature_schema`, lines 593-669)
- Why fragile: Adding any new edge feature requires updating all four locations in the same order. The original critical bug (see `BUGFIXES.md`) was caused by exactly this pattern. The `FeatureSchema` class exists in `data.py` but is not used outside of data processing; `ar_utils.get_edge_feature_indices` and `masking.py` both reimplement the same ordering logic independently.
- Safe modification: Always add new edge features in ALL four locations in the same relative position. Verify with the existing feature schema tests in `tests/test_feature_schema.py`.
- Test coverage: `tests/test_feature_schema.py` exists. Ensure it covers all currently enabled feature combinations.

**`check_puzzle_solved` in `utils.py` does not verify crossing constraints:**
- Files: `src/hashi_puzzle_solver/utils.py` lines 17-61
- Why fragile: The function is used during AR rollout to detect completion. It checks degree constraints and bridge-count limits but explicitly skips crossing validation with the comment "simplified - assume bridges are properly placed". A model that predicts crossing bridges will be incorrectly marked as solved.
- Safe modification: Do not rely on `check_puzzle_solved` for correctness guarantees. Use it only as a quick pre-filter; follow up with the full constraint check in `ar_engine.py` which uses `edge_conflict_index`.
- Test coverage: No dedicated test for the crossing constraint omission.

**`DiffusionTrainer` carry-over buffer leaks state between train/val within an epoch:**
- Files: `src/hashi_puzzle_solver/diffusion_engine.py` lines 48-51, 138-236
- Why fragile: The `carry_over_buffer_train` and `carry_over_buffer_val` are instance variables that persist across the full training run. If `run_epoch` is called out of order (e.g., val before train at epoch 1, or two val calls), the buffers contain stale puzzle states from a different epoch's training pass. The code comment acknowledges the concern ("Separate buffers for train and validation to avoid data leakage") but the buffer state is never reset between epochs.
- Safe modification: Add `recursive_carryover: false` in configs where buffer correctness cannot be verified (the active `diffusion_solver_continuous.yaml` already sets this to `false`).

**`redistribute_edge_conflicts` in `ar_engine.py` may silently produce empty conflicts:**
- Files: `src/hashi_puzzle_solver/ar_engine.py` lines 25-42
- Why fragile: After `Batch.to_data_list()`, `edge_conflict_index` may be a zero-dimensional tensor or absent. The function converts it back to `edge_conflicts` list using `.t().tolist()`, but if PyG's slice dictionary is corrupted (e.g., after a batch subset operation), the reconstructed conflicts may be wrong or empty with no error raised.
- Safe modification: Add an assertion comparing the sum of `edge_conflicts` list lengths to the original batch's `edge_conflict_index` column count after redistribution.

---

## Scaling Limits

**75,000 raw JSON files in a flat directory:**
- Current capacity: 75,000 files verified. `dataset/raw/` contains `puzzle_00000000.json` through `puzzle_00074999.json` in a single flat directory.
- Limit: Most filesystems (ext4, APFS) handle this, but `glob("puzzle_*.json")` order is non-deterministic on some systems without explicit sort, and `_get_filtered_filenames` must open all 75,000 files on startup. At ~500,000 files the startup scan would take tens of seconds.
- Scaling path: Implement an index CSV (see performance section above), and partition raw files into subdirectories by split.

**MLflow tracking uses a local SQLite database:**
- Current capacity: Single `mlflow.db` file in the project root. Concurrent writes from parallel tuning trials will cause SQLite lock contention.
- Limit: `optuna`-based hyperparameter tuning in `src/hashi_puzzle_solver/tune.py` runs trials sequentially by default, but parallel tuning is possible.
- Scaling path: Switch to a PostgreSQL or MLflow server backend for parallel experiment tracking.

---

## Dependencies at Risk

**`hashi` package (version `>=0.1.0`):**
- Risk: Listed as a dependency in `pyproject.toml` and used in the committed-but-broken `yo2.py`. This package is not a standard PyPI package and has no version pinning beyond `>=0.1.0`. Its API may change without notice.
- Impact: `yo2.py` already uses a stale import from `src.bridges_gen` (wrong path); if `hashi` is updated, no tests will catch breakage.
- Migration plan: Remove `yo2.py`; audit any real usage of `hashi` in non-scratch code and pin the version in `pyproject.toml`.

**`scipy.sparse.linalg.eigsh` with `maxiter=1000`:**
- Risk: The `eigsh` API has changed across scipy versions. `maxiter` semantics differ between sparse and dense backends. The current `requirements.txt` or `pyproject.toml` pins `scipy` indirectly through `torch-geometric`.
- Impact: Silent fall-back to `[0.0, 0.0, 0.0]` spectral features on convergence failure means the model may train with degraded features and no warning logged.
- Migration plan: Pin scipy explicitly; emit a warning with puzzle ID on convergence failure.

---

## Missing Critical Features

**No test split in the current dataset / training pipeline:**
- Problem: The `HashiDataset` accepts `split="test"`, and raw files can have `"split": "test"`, but no config, trainer, or evaluation script uses the test split. All reported numbers are on the validation set.
- Blocks: True generalization measurement; comparison with other puzzle-solving benchmarks.

**`check_puzzle_solved` does not verify crossing constraints:**
- Problem: The connectivity graph used during AR rollout completion detection ignores bridge-crossing violations (see Fragile Areas above).
- Blocks: Reporting of true "valid solution" rate; the reported perfect-puzzle accuracy may be inflated for models that produce crossing bridges.

---

## Test Coverage Gaps

**`diffusion_engine.py` carry-over buffer logic (`_prepare_mixed_batch`, `_refill_buffer`):**
- What's not tested: The interaction between `carry_over_buffer_train` and `carry_over_buffer_val` across multiple epochs; buffer overflow truncation; student-forcing stepping behavior.
- Files: `src/hashi_puzzle_solver/diffusion_engine.py` lines 53-236
- Risk: Silent data leakage from train to val buffer, or stale puzzles accumulating in the buffer.
- Priority: High

**`engine.py` `SubsetRandomSampler` validation subset selection:**
- What's not tested: That the val subset is reproducible, representative, and correctly sized.
- Files: `src/hashi_puzzle_solver/engine.py` lines 514-529
- Risk: Validation metrics are silently biased to the first N sorted files.
- Priority: High

**`check_puzzle_solved` crossing constraint omission:**
- What's not tested: A puzzle with crossing bridges is correctly identified as unsolved.
- Files: `src/hashi_puzzle_solver/utils.py` lines 17-61
- Risk: Inflated perfect-puzzle accuracy during AR rollout evaluation.
- Priority: Medium

**`n_blocks` + `num_inference_steps_training = 1` division by zero:**
- What's not tested: Config validation or defensive guard against this configuration.
- Files: `src/hashi_puzzle_solver/diffusion_engine.py` line 595
- Risk: Silent crash in production if config is changed.
- Priority: Medium

**`src2` trainers (diffusion, AR, one-shot) parity with `src`:**
- What's not tested: `tests_src2/test_parity.py` exists but its scope is unclear. Full functional equivalence between `src` and `src2` diffusion training loops is not verified.
- Files: `src2/hashi_puzzle_solver/trainers/diffusion.py`, `src2/hashi_puzzle_solver/trainers/ar.py`
- Risk: `src2` silently diverges from `src` behavior as both evolve.
- Priority: Medium (lower if `src2` becomes canonical)

---

*Concerns audit: 2026-03-06*
