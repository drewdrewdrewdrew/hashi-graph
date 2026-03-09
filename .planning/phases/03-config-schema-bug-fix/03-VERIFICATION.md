---
phase: 03-config-schema-bug-fix
verified: 2026-03-09T00:00:00Z
status: passed
score: 6/6 must-haves verified
re_verification: false
---

# Phase 3: Config Schema + Bug Fix Verification Report

**Phase Goal:** Config types for both new components exist in `config.py`, `ModelConfig` has typed fields for both, a reference YAML is in place, and the `scales` crash is eliminated — so no implementation code can reference `reasoning` or `reverse_gnn` without a typed home
**Verified:** 2026-03-09
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `ReasoningConfig` and `ReverseGnnConfig` dataclasses exist in `config.py` with correct fields and defaults | VERIFIED | Both classes present at lines 132-149; `ReasoningConfig(enabled=False, steps=5)` with `__post_init__` validation; `ReverseGnnConfig(enabled=False, separate_weights=True, project_embeddings=True)` with no validation |
| 2 | `ModelConfig` has typed `reasoning` and `reverse_gnn` fields, both defaulting to disabled | VERIFIED | Lines 109-110: `reasoning: "ReasoningConfig" = field(default_factory=lambda: ReasoningConfig())` and `reverse_gnn: "ReverseGnnConfig" = field(default_factory=lambda: ReverseGnnConfig())` |
| 3 | `HashiModelConfig.from_dict` correctly parses `reasoning` and `reverse_gnn` sub-dicts from `model_dict` | VERIFIED | Lines 245-253: extracts sub-dicts, builds `model_base_dict` filtering both keys, passes typed instances to `ModelConfig(...)` |
| 4 | Existing YAML configs load without error after `ModelConfig` changes | VERIFIED | `diffusion_solver_continuous.yaml` and `diffusion_solver_continuous_bptt.yaml` both load cleanly via smoke test; 10/10 TDD tests pass |
| 5 | `rev_reasoning.yaml` exists with `training.mode: rev-reason` and both model config blocks | VERIFIED | File at `configs/rev_reasoning.yaml` line 100: `mode: "rev-reason"`; lines 87-95: `reasoning` and `reverse_gnn` blocks with correct defaults; diffusion params commented out lines 115-124 |
| 6 | Running any training mode alongside `bptt.enabled: true` no longer crashes with `UnboundLocalError: scales` | VERIFIED | `scales = None` at line 354 of `diffusion.py`, immediately after `batch = batch.to(self.device)` (line 353) and before `if mode == "diff-cont":` (line 356) — all non-diff-cont paths now have `scales` defined |

**Score:** 6/6 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src2/hashi_puzzle_solver/models/config.py` | `ReasoningConfig`, `ReverseGnnConfig` dataclasses; updated `ModelConfig` and `from_dict` | VERIFIED | All four additions present and substantive; wired via `from_dict` and `ModelConfig` field declarations |
| `src2/hashi_puzzle_solver/trainers/diffusion.py` | `scales = None` initialization before mode branch | VERIFIED | Line 354: `scales = None` is the second statement in the batch loop body, preceding the mode dispatch |
| `configs/rev_reasoning.yaml` | Reference config for rev-reason training mode | VERIFIED | File exists, 147 lines, loads correctly via `HashiModelConfig.from_yaml`, all asserted field values confirmed |
| `tests/test_config_reasoning.py` | 10 TDD tests covering all config behavior | VERIFIED | All 10 tests pass (10 passed, 0 failed, 5.07s); tests cover defaults, validation, `ModelConfig` fields, `from_dict` parsing, and YAML backward compat |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `HashiModelConfig.from_dict` | `ModelConfig` | `reasoning` and `reverse_gnn` sub-dicts stripped from `model_dict` before `**model_base_dict` splat | VERIFIED | Lines 245-253: `reasoning_dict = model_dict.get("reasoning", {})`, `reverse_gnn_dict = model_dict.get("reverse_gnn", {})`, list comprehension excludes both from `model_base_dict`, then `ModelConfig(**model_base_dict, reasoning=ReasoningConfig(**reasoning_dict), reverse_gnn=ReverseGnnConfig(**reverse_gnn_dict))` |
| `configs/rev_reasoning.yaml` | `HashiModelConfig.from_dict` | `model.reasoning` and `model.reverse_gnn` YAML blocks map to typed sub-config instances | VERIFIED | YAML blocks at lines 87-95 parse correctly; smoke test confirms `cfg.model.reasoning.enabled == False`, `steps == 5`, `cfg.model.reverse_gnn.separate_weights == True`, `project_embeddings == True` |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| BUG-01 | 03-01-PLAN.md | BPTT can be enabled alongside any training mode without a `scales` UnboundLocalError crash | SATISFIED | `scales = None` at diffusion.py:354; `_run_bptt_window` at line 192 accepts `scales` param; bptt dispatch path at lines 527-563 passes `scales` to `_run_bptt_window` — `scales` is always defined when consumed |
| CFG-05 | 03-01-PLAN.md | `ReasoningConfig` dataclass with `enabled: bool = False`, `steps: int = 5`, `steps >= 1` validation | SATISFIED | config.py lines 132-140; test_reasoning_config_steps_zero_raises PASSED |
| CFG-06 | 03-01-PLAN.md | `ReverseGnnConfig` dataclass with correct field defaults | SATISFIED | config.py lines 143-149; test_reverse_gnn_config_defaults PASSED |
| CFG-07 | 03-01-PLAN.md | `ModelConfig` has typed `reasoning` and `reverse_gnn` fields, both defaulting to disabled | SATISFIED | config.py lines 109-110; test_model_config_has_reasoning_field and test_model_config_has_reverse_gnn_field PASSED |
| CFG-08 | 03-01-PLAN.md | New `rev_reasoning.yaml` with correct structure | SATISFIED | File exists; `training.mode: rev-reason`; both model blocks present; diffusion params commented out; loads via `from_yaml` without error |

**Orphaned requirements check:** REQUIREMENTS.md traceability table maps BUG-01, CFG-05, CFG-06, CFG-07, CFG-08 exclusively to Phase 3. All five are claimed in 03-01-PLAN.md. No orphaned requirements.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | None found |

No TODOs, FIXMEs, placeholders, empty implementations, or stub returns found in any of the four files touched by this phase.

**Forward-reference note (informational):** `ModelConfig` uses `field(default_factory=lambda: ReasoningConfig())` rather than `field(default_factory=ReasoningConfig)`. This is the correct approach given that `ReasoningConfig` and `ReverseGnnConfig` are defined after `ModelConfig` in the source file. The lambda defers class lookup until instantiation time. No issue.

---

### Human Verification Required

None. All phase 3 deliverables are statically verifiable:
- Dataclass fields and defaults: verified by running the test suite
- YAML loading: verified by smoke test
- `scales = None` placement: verified by grep showing line ordering (353 batch move, 354 scales init, 356 mode branch)
- Git commits: all three documented hashes (`3cdd142`, `844ad86`, `0642569`) confirmed present in git log

---

### Summary

Phase 3 achieved its goal completely. All six observable truths hold, all four artifacts exist with substantive content and correct wiring, all five requirement IDs are satisfied with code evidence, and the test suite passes 10/10 with no regressions. The `scales` crash path is closed by a one-line initializer in exactly the right position. The typed config foundation is in place for Phase 4 plans to reference `cfg.model.reasoning` and `cfg.model.reverse_gnn` without TypeError.

---

_Verified: 2026-03-09_
_Verifier: Claude (gsd-verifier)_
