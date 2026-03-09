---
phase: 02-bptt-training-loop
verified: 2026-03-06T15:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
gaps: []
human_verification: []
---

# Phase 2: BPTT Training Loop Verification Report

**Phase Goal:** When `bptt.enabled: true`, the diffusion training loop uses sliding-window backpropagation through time, with gradient checkpointing and window-loss EMA, while remaining byte-for-byte equivalent to current behavior when disabled.
**Verified:** 2026-03-06T15:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1 | When `bptt.enabled=false`, training loop produces identical loss/optimizer behavior — `torch.stack(step_losses).mean().backward()` then `optimizer.step()` | VERIFIED | Lines 519-524 of diffusion.py: `if not bptt_enabled:` branch calls `total_batch_loss.backward()` + `self.optimizer.step()` unchanged. Tests `test_bptt_disabled_backward_called` and `test_bptt_disabled_zero_grad_called` pass. |
| 2 | When `bptt.enabled=true`, the sliding window iterates over `[0, stride, 2*stride, ...]` and each window covers `[start, min(start+window, num_steps))` | VERIFIED | Lines 533-538: `window_starts = list(range(0, num_inference_steps_training, bptt_stride))` and `w_end = min(w_start + bptt_window, num_inference_steps_training)`. Tests `test_bptt_window_count_stride1` (3 windows) and `test_bptt_window_count_stride2` (2 windows) pass. |
| 3 | Within each window, gradient flows through step transitions without `torch.no_grad()` and without `.detach()` on the transition tensor | VERIFIED | `_run_bptt_window` at lines 286-306: transition block executes `probs -> probs_centered -> target_state` with no `.detach()` call on `target_state`, outside any `no_grad` context. |
| 4 | `torch.utils.checkpoint.checkpoint` is applied per step inside `_run_bptt_window` | VERIFIED | Line 7: `from torch.utils.checkpoint import checkpoint`. Lines 279-283: `step_loss, logits = checkpoint(_step_forward, edge_attr_input, use_reentrant=False)`. Test `test_window_loss_backward_does_not_raise` confirms `.backward()` succeeds. |
| 5 | Each window calls `.backward()` before the next window starts; gradients accumulate across overlapping windows; `optimizer.step()` fires exactly once per batch | VERIFIED | Lines 550-561: `window_loss.backward(retain_graph=more_windows)` inside the window loop; `self.optimizer.step()` after the loop. Tests `test_bptt_window_count_stride1` and `test_bptt_window_count_stride2` verify `optimizer.step.call_count == 1`. |
| 6 | Loss reported is the EMA-smoothed mean of window-averaged step losses (`ema = decay * ema + (1-decay) * window_loss.item()`) | VERIFIED | Lines 555-558: `bptt_ema = wl` on first window; `bptt_ema = bptt_decay * bptt_ema + (1.0 - bptt_decay) * wl` on subsequent. Line 562: `total_batch_loss_value = bptt_ema`. Test `test_bptt_ema_updates` verifies `result["loss"]` is a positive float. |
| 7 | When `training=False`, `bptt_enabled` is False regardless of config — eval path uses the existing `no_grad` path | VERIFIED | Line 342: `bptt_enabled = (... _bptt.get("enabled", False) ...) and training`. Test `test_bptt_eval_path_unaffected` passes with `optimizer.step.call_count == 0`. |
| 8 | Edge-logit state at each step boundary is stored as detached clone in `step_boundary_states` | VERIFIED | Lines 476-478: `if bptt_enabled and self.bridge_logits_idx is not None: _logit_slice = current_data.edge_attr[..].detach().clone(); step_boundary_states.append(_logit_slice)`. Test `test_bptt_enabled_populates_state_cache` verifies shape, count, and `requires_grad=False`. |
| 9 | `_run_bptt_window` returns a differentiable scalar; `.backward()` yields non-None `.grad` on model parameters | VERIFIED | `return torch.stack(window_losses).mean()` at line 308. Tests `test_run_bptt_window_returns_scalar` (0-dim), `test_run_bptt_window_has_grad` (requires_grad=True), and `test_window_loss_backward_does_not_raise` (param.grad not None) all pass. |

**Score:** 9/9 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src2/hashi_puzzle_solver/trainers/diffusion.py` | BPTT-aware `run_epoch` with disabled-path guard, step-state cache, `_run_bptt_window`, window loop, EMA | VERIFIED | Contains `bptt_enabled`, `step_boundary_states`, `_run_bptt_window` method, sliding-window loop, `bptt_ema`. 697 lines total, substantive. |
| `src2/hashi_puzzle_solver/trainers/test_diffusion_compat.py` | Backward-compat regression tests containing `test_bptt_disabled_identical` (or equivalent) | VERIFIED | Contains 6 tests across 4 test classes. Pattern `test_bptt_disabled_backward_called` exists. All 6 pass. |
| `src2/hashi_puzzle_solver/trainers/test_diffusion_bptt.py` | BPTT window loop tests containing `test_bptt_window_backward` (or equivalent) | VERIFIED | Contains 7 tests across 7 test classes covering scalar return, grad, backward, stride-1, stride-2, EMA, single-window. All 7 pass. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `run_epoch` bptt dispatch | `training_cfg.bptt.enabled` | `bptt_enabled = (_bptt.get("enabled", False) if isinstance(_bptt, dict) else _bptt.enabled) and training` | WIRED | Line 342. Pattern `bptt_enabled` used at lines 476 and 519. Note: plan key_link pattern was `bptt_cfg\.enabled`; actual variable is `bptt_enabled` (inline extraction). Functionally equivalent — not a gap. |
| `step_boundary_states` list | each train step's logit output | `step_boundary_states.append(_logit_slice)` at line 478 | WIRED | `step_boundary_states.append` confirmed at line 478, guarded by `bptt_enabled and self.bridge_logits_idx is not None`. |
| `run_epoch` bptt_enabled=True branch | `_run_bptt_window` | called once per window with `step_boundary_states[start]` | WIRED | Line 539: `window_loss = self._run_bptt_window(start_data=data, start_step=w_start, ...)`. |
| `_run_bptt_window` | `torch.utils.checkpoint.checkpoint` | wraps `_step_forward` per step | WIRED | Line 279: `step_loss, logits = checkpoint(_step_forward, edge_attr_input, use_reentrant=False)`. Note: plan pattern was `checkpoint\.checkpoint`; actual call is `checkpoint(...)` (imported directly). Functionally equivalent — not a gap. |
| window loss | `bptt_ema` scalar | `ema = decay * ema + (1-decay) * window_loss.item()` | WIRED | Lines 554-558: `wl = window_loss.item()` then EMA update. `total_batch_loss_value = bptt_ema` at line 562. |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| TRN-01 | 02-01 | When `bptt.enabled`, forward pass caches edge logits at each step boundary | SATISFIED | Lines 476-478: `step_boundary_states.append(detached clone)` per inference step. |
| TRN-02 | 02-02 | Sliding window iterates over step sequence with configured window + stride | SATISFIED | Lines 533-538: `range(0, num_inference_steps_training, bptt_stride)`, `min(w_start + bptt_window, ...)`. |
| TRN-03 | 02-02 | Within each window, gradient flows through consecutive step transitions (no `no_grad` block) | SATISFIED | `_run_bptt_window` executes step transitions outside `no_grad`, no `.detach()` on `target_state`. |
| TRN-04 | 02-02 | Gradient checkpointing applied within each window to bound activation memory | SATISFIED | `checkpoint(_step_forward, edge_attr_input, use_reentrant=False)` at line 279. |
| TRN-05 | 02-02 | Gradients accumulate across overlapping windows before optimizer step | SATISFIED | `window_loss.backward(retain_graph=more_windows)` per window; single `self.optimizer.step()` after all windows. |
| TRN-06 | 02-02 | Loss averaged across steps within each window; EMA applied to that scalar | SATISFIED | `_run_bptt_window` returns `torch.stack(window_losses).mean()`. EMA update in `run_epoch` lines 554-562. |
| COMP-01 | 02-01 | When `bptt.enabled: false`, training loop is byte-for-byte equivalent to current behavior | SATISFIED | `if not bptt_enabled:` branch at lines 519-524 is structurally identical to the pre-BPTT code. `test_bptt_disabled_backward_called` and `test_bptt_eval_path_unaffected` pass. |
| COMP-02 | 02-01 | All existing config fields (`num_inference_steps_training`, `n_blocks`, etc.) remain valid | SATISFIED | Lines 348-349: both fields read from `training_cfg` via `.get()` as before. `n_blocks` subsampling block at lines 504-517 is unchanged. |

**Orphaned requirements check:** REQUIREMENTS.md maps exactly {TRN-01, TRN-02, TRN-03, TRN-04, TRN-05, TRN-06, COMP-01, COMP-02} to Phase 2. All 8 are claimed by plans 02-01 and 02-02 and verified above. No orphans.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No anti-patterns found |

Scan coverage: `diffusion.py`, `test_diffusion_compat.py`, `test_diffusion_bptt.py`. No TODO/FIXME/PLACEHOLDER/console.log/return null/empty implementation patterns. The previous `NotImplementedError` stub is confirmed removed (grep returned no matches for `NotImplementedError` or `raise Not`).

---

## Human Verification Required

None. All goal-critical behaviors are mechanically verifiable:
- Optimizer call counts: verified via mock call_count assertions in tests
- Gradient flow: verified via `param.grad is not None` assertion
- Checkpoint import and call: verified via grep
- EMA formula: verified via code inspection and `test_bptt_ema_updates`
- Eval guard: verified via `test_bptt_eval_path_unaffected`

The only behaviors that would typically need human verification (visual, real-time, external services) are not applicable to a training loop refactor.

---

## Commit Verification

All 4 commits documented in SUMMARYs exist in the repository:

| Commit | Type | Description |
|--------|------|-------------|
| `d188836` | test(RED) | Failing tests for BPTT dispatch |
| `2abe3ee` | feat(GREEN) | BPTT dispatch + step-state cache |
| `7b2c883` | feat | `_run_bptt_window` with gradient checkpointing |
| `e3487df` | feat | BPTT window loop + EMA wired into `run_epoch` |

---

## Summary

Phase 2 goal is fully achieved. All 9 observable truths are verified against the actual codebase, all 3 artifacts are substantive and wired, all 5 key links are confirmed, and all 8 requirements (TRN-01 through TRN-06, COMP-01, COMP-02) are satisfied.

The full test suite (13 tests across 2 files) passes with 0 failures.

Two minor documentation discrepancies in plan key_link `pattern` fields:
- Plan 02-01 specified `bptt_cfg\.enabled` but the code uses the computed variable `bptt_enabled`. The config access is present and correct.
- Plan 02-02 specified `checkpoint\.checkpoint` but the code imports `checkpoint` directly and calls `checkpoint(...)`. The import and call are present and correct.

Neither is a functional gap.

---

_Verified: 2026-03-06T15:00:00Z_
_Verifier: Claude (gsd-verifier)_
