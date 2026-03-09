---
phase: 5
slug: integration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none (auto-discovered) |
| **Quick run command** | `pytest tests/test_hashi_graph_model_integration.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_hashi_graph_model_integration.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 5-01-01 | 01 | 0 | SC-1,2,3,4 | unit | `pytest tests/test_hashi_graph_model_integration.py -x -q` | ❌ W0 | ⬜ pending |
| 5-01-02 | 01 | 1 | SC-2 | unit | `pytest tests/test_hashi_graph_model_integration.py::test_flags_disabled_baseline -x -q` | ❌ W0 | ⬜ pending |
| 5-01-03 | 01 | 1 | SC-4 | unit | `pytest tests/test_hashi_graph_model_integration.py::test_edge_head_dim_all_combos -x -q` | ❌ W0 | ⬜ pending |
| 5-01-04 | 01 | 1 | SC-1 | unit | `pytest tests/test_hashi_graph_model_integration.py::test_both_flags_enabled -x -q` | ❌ W0 | ⬜ pending |
| 5-01-05 | 01 | 1 | SC-3 | unit | `pytest tests/test_hashi_graph_model_integration.py::test_rev_reasoning_interleaved -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_hashi_graph_model_integration.py` — stubs for all 4 success criteria:
  - `test_flags_disabled_baseline` — byte-for-byte equivalence (SC-2)
  - `test_reasoning_only` — IterativeBackbone wired, output shape correct (SC-1)
  - `test_reverse_only_with_projection` — ReverseBackbone concat + projection, correct EdgeHead dim (SC-4)
  - `test_reverse_only_no_projection` — `2 * final_dim` passed to EdgeHead, no shape error (SC-4)
  - `test_both_flags_enabled` — rev-reasoning composition, shapes correct (SC-1)
  - `test_rev_reasoning_interleaved` — K iterations each using fwd + rev + residual (SC-3)
  - `test_edge_head_dim_all_combos` — 4 flag combinations produce no shape mismatch (SC-4)

---

## Manual-Only Verifications

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
