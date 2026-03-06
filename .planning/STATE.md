# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-06)

**Core value:** The model learns multi-step coordination by receiving gradient signal that flows across consecutive diffusion steps
**Current focus:** Phase 1 — Config Schema

## Current Position

Phase: 1 of 2 (Config Schema)
Plan: 1 of 1 in current phase
Status: In progress
Last activity: 2026-03-06 — Completed 01-01 (BpttConfig schema)

Progress: [█░░░░░░░░░] 10%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 4 min
- Total execution time: 0.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-config-schema | 1 | 4 min | 4 min |

**Recent Trend:**
- Last 5 plans: 4 min
- Trend: baseline

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Sliding window chosen over full-sequence BPTT (memory constraint; pending outcome)
- Gradient checkpointing chosen over detach-at-boundary (memory over speed; pending outcome)
- Gradient accumulation across overlapping windows (steps get signal from all covering windows; pending outcome)
- EMA on window loss scalar only, not model weights (stabilize loss signal; pending outcome)
- [01-01] BpttConfig placed before LossWeightsConfig (alphabetical grouping of nested configs)
- [01-01] loss_ema_decay uses half-open interval [0, 1) — value 1.0 excluded (non-decaying EMA)
- [01-01] bptt defaults to enabled=False so existing configs load without modification

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-06
Stopped at: Completed 01-01-PLAN.md (BpttConfig schema)
Resume file: None
