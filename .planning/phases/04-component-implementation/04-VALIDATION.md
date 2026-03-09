---
phase: 4
slug: component-implementation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` with `testpaths = ["tests"]`, `pythonpath = ["src2"]` |
| **Quick run command** | `pytest tests/test_iterative_backbone.py tests/test_reverse_backbone.py tests/test_diffusion_rev_reason.py -x -q` |
| **Full suite command** | `pytest tests/ tests_src2/ -x -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_iterative_backbone.py tests/test_reverse_backbone.py tests/test_diffusion_rev_reason.py -x -q`
- **After every plan wave:** Run `pytest tests/ tests_src2/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-01-W0 | 01 | 0 | MODE-01, MODE-02 | unit | `pytest tests/test_diffusion_rev_reason.py -x -q` | ❌ W0 | ⬜ pending |
| 4-01-01 | 01 | 1 | MODE-01 | unit | `pytest tests/test_diffusion_rev_reason.py::test_rev_reason_no_noise_injection -x` | ❌ W0 | ⬜ pending |
| 4-01-02 | 01 | 1 | MODE-02 | unit | `pytest tests/test_diffusion_rev_reason.py::test_rev_reason_component_flags_independent -x` | ❌ W0 | ⬜ pending |
| 4-02-W0 | 02 | 0 | REAS-01, REAS-02 | unit | `pytest tests/test_iterative_backbone.py -x -q` | ❌ W0 | ⬜ pending |
| 4-02-01 | 02 | 1 | REAS-01 | unit | `pytest tests/test_iterative_backbone.py::test_iterative_backbone_applies_k_times -x` | ❌ W0 | ⬜ pending |
| 4-02-02 | 02 | 1 | REAS-02 | unit | `pytest tests/test_iterative_backbone.py::test_iterative_backbone_steps_parameter -x` | ❌ W0 | ⬜ pending |
| 4-02-03 | 02 | 1 | REAS-01+02 | unit | `pytest tests/test_iterative_backbone.py::test_steps_one_matches_single_pass -x` | ❌ W0 | ⬜ pending |
| 4-03-W0 | 03 | 0 | REVG-01, REVG-02, REVG-03 | unit | `pytest tests/test_reverse_backbone.py -x -q` | ❌ W0 | ⬜ pending |
| 4-03-01 | 03 | 1 | REVG-01 | unit | `pytest tests/test_reverse_backbone.py::test_reverse_backbone_output_shape -x` | ❌ W0 | ⬜ pending |
| 4-03-02 | 03 | 1 | REVG-02 | unit | `pytest tests/test_reverse_backbone.py::test_separate_weights_independence -x` | ❌ W0 | ⬜ pending |
| 4-03-03 | 03 | 1 | REVG-03 | unit | `pytest tests/test_reverse_backbone.py::test_project_embeddings_output_dim -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_iterative_backbone.py` — stubs for REAS-01, REAS-02 (steps=1 identity, K-iteration residual)
- [ ] `tests/test_reverse_backbone.py` — stubs for REVG-01, REVG-02, REVG-03 (shape, weight sharing, projection)
- [ ] `tests/test_diffusion_rev_reason.py` — stubs for MODE-01, MODE-02 (no noise injection, independent flags)

*No framework install needed — pytest already configured in pyproject.toml.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | — | — | — |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
