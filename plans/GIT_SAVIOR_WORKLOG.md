# Step 0 — Baseline Inventory

_Generated: 2026-03-03_

---

## Stash Summary

| Ref | Branch | Message |
|-----|--------|---------|
| `stash@{0}` | (no branch) | `better rollout sampling` |
| `stash@{1}` | `refactor` | `granular emb dim size, add noise emb to edges` |
| `stash@{2}` | `refactor` | `hierarchical-features-plus-h1-h3-fixes` |
| `stash@{3}` | `memory-optimizations` | `add enhancements` |

### Overlap Confirmation: stash@{1} vs stash@{2}

Both stashes contain **exactly the same 38 files** with identical change counts
(`2373 insertions / 724 deletions`). The only difference is binary (`mlflow.db`
size). All code changes are identical — `stash@{2}` is a snapshot of `stash@{1}`
taken on the `refactor` branch.

---

## Bug-Fix Hunks Identified

### `src2/hashi_puzzle_solver/trainers/base.py` (+75 lines)

#### Hunk 1 — Split-specific dataset limits  _(Step 4 config support)_

```diff
-        limit = data_config.get("limit")
+        legacy_limit = data_config.get("limit")
+        split_limit = (
+            data_config.get("train_limit") if split == "train"
+            else data_config.get("val_limit")
+        )
+        limit = split_limit if split_limit is not None else legacy_limit
```

Adds `train_limit` / `val_limit` keys with fallback to legacy `limit`.
**Status: APPLIED in this session (commit `10fdd01 config: add split limits and val sampler seed`).**

Config keys added to `configs/diffusion_solver_continuous.yaml`:
- `data.train_limit: 1500`
- `data.val_limit: 500`
- `data.val_sampler_seed: 42`

All 12 tests green (dataloader sampling + diffusion suite).

#### Hunk 2 — val_sampler_seed + randperm  _(Step 1)_

```diff
 elif split == "val":
     from torch.utils.data import SubsetRandomSampler
-    indices = list(range(num_samples))
+    val_sampler_seed = int(data_config.get("val_sampler_seed", 42))
+    generator = torch.Generator().manual_seed(val_sampler_seed)
+    indices = torch.randperm(
+        len(dataset),
+        generator=generator,
+    )[:num_samples].tolist()
     sampler = SubsetRandomSampler(indices)
     shuffle = False
```

Replaces deterministic sequential slice `[0..N]` with a seeded random permutation.
Ensures the val subset is reproducible but non-trivial (avoids always picking the
first N samples).  
**Status: APPLIED in this session (commit `fix(trainers): randomize val sampler with seed`).**

#### Hunk 3 — model_config attribute migration  _(refactor, not a bug fix)_

The stash replaces 20+ `model_config.get(...)` call-site kwargs with typed
`self.model_config.model.<field>` attribute access. This is a refactoring hunk,
not a bug fix, and is intentionally deferred.

---

### `src2/hashi_puzzle_solver/trainers/diffusion.py` (+404 lines)

This file has two isolated bug-fix hunks and a large hierarchical-feature block.

#### Hunk 1 — sigma_max for zero_mask  _(Step 2)_

```diff
+        sigma_max = training_cfg.get("sigma_max", 2.0)
         fresh_alphas = torch.zeros(n_fresh, device=self.device)
         fresh_sigmas = torch.full((n_fresh,), sigma_max, device=self.device)
```

Previously `alpha_power` was used to sample non-zero alphas; stash replaces with
hard zeros and moves `sigma_max` lookup to just-in-time position.  
**Status: APPLIED in this session (commit `f7be45a fix(diffusion): enforce sigma_max for zero_mask`).**

Tests added: `test_fresh_alphas_are_zero`, `test_fresh_sigmas_equal_sigma_max`
in `tests_src2/test_diffusion.py` — both passing (5/5 suite green).

#### Hunk 2 — noise_pred dict handling + n_blocks slice  _(Step 3)_

```diff
         if noise_pred is not None:
+            if isinstance(noise_pred, dict):
+                noise_pred = noise_pred["global"].detach()
+            else:
                 noise_pred = noise_pred.detach()
```

Carry-over buffer was crashing when model returned a `dict` for hierarchical mode.
Applied in two places:
- `_refill_buffer`: dict guard before indexing per-graph noise.
- `run_epoch` carry-over assignment (`current_input_noise = noise_pred.detach()`): extract
  `noise_pred["global"]` so the tensor survives the `indices`-slice that follows.  
**Status: APPLIED in this session (commit `03c349b fix(diffusion): slice noise preds on subsample`).**

Tests added: `test_refill_buffer_tensor_noise_pred`, `test_refill_buffer_dict_noise_pred`,
`test_refill_buffer_dict_noise_pred_values` in `tests_src2/test_diffusion.py` — all 8/8 suite green.

#### Large block — Hierarchical features (progress conditioning, boundary flags,
component noise, rewired edges)

~350-line addition gating on `use_hierarchical_noise_prediction`,
`use_progress_conditioning`, `use_boundary_flag`. All gated to `False` by default
in the config — zero runtime impact when disabled.  
**Status: not yet applied — belongs to Step 6 (feature branch).**

---

## Config-Only Changes (`configs/diffusion_solver_continuous.yaml`)

Changes scoped to Step 4:

| Key | Section | Value | Purpose |
|-----|---------|-------|---------|
| `train_limit` | `data` | `1500` | Cap training split size |
| `val_limit` | `data` | `500` | Cap val split to random 500 |
| `val_sampler_seed` | `data` | `42` | Seed for val subset reproducibility |
| `use_logit_embeddings` | `model` | `false` | New optional embedding toggle |
| `noise_embedding_dim` | `model` | `16` | Noise projection dim |
| `use_noise_in_message_passing` | `model` | `true` | Concat noise to GNN edge features |
| `use_noise_in_prediction` | `model` | `false` | Concat noise to EdgeHead |
| `use_noise_in_global_meta` | `model` | `true` | Add noise to global meta node |
| `use_progress_conditioning` | `model` | `false` | Prophet head injection (gated) |
| `use_boundary_flag` | `model` | `false` | Cross-component edge flag (gated) |
| `rollout_init_mode` | `training` | `random` | Control rollout initialisation |

**Status (split-limit + seed keys): APPLIED in this session (commit `10fdd01 config: add split limits and val sampler seed`).**  
Note: the noise-embedding and hierarchical toggles (`use_logit_embeddings`, `noise_embedding_dim`, `use_noise_in_*`, `use_progress_conditioning`, `use_boundary_flag`, `rollout_init_mode`) remain deferred to Step 6 (feature branch).

---

## Reversibility Notes

- All three bug-fix hunks in `base.py` and `diffusion.py` are single-concern and
  independently revertable.
- Hierarchical feature hunks in `diffusion.py` are fully gated by config flags
  defaulting to `false` — they can be landed without affecting existing runs.
- No commit in this sequence touches more than one logical concern.
