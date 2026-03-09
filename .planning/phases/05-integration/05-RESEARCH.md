# Phase 5: Integration - Research

**Researched:** 2026-03-09 (updated 2026-03-09 with as-built verification)
**Domain:** PyTorch nn.Module composition — wiring IterativeBackbone and ReverseBackbone into HashiGraphModel.forward(), EdgeHead dimension adaptation
**Confidence:** HIGH

## Summary

Phase 5 is a pure in-codebase integration task. No new libraries are needed. All components were delivered by Phase 4 and are independently tested. The PLAN (05-integration-01-PLAN.md) already exists and is accurate. This research document has been updated with as-built verification confirming all component contracts match what the plan expects.

The work is:

1. **HashiGraphModel.__init__**: Add two optional args (`iterative_backbone`, `reverse_backbone`); store via normal `self.x = x` assignment; validate `project_embeddings=True` constraint when both are enabled.
2. **HashiGraphModel.forward() step 6**: Replace the single `self.backbone(h, ...)` call with a three-branch composition block: interleaved loop (both enabled), sequential IterativeBackbone only, sequential ReverseBackbone only, or pass-through (neither).
3. **ModelFactory.create_model**: Construct optional components from config flags; compute `edge_head_node_dim` before building EdgeHead; update ProphetHead dim too; pass new components into HashiGraphModel constructor.
4. **Backward-compatibility guard**: With all flags disabled, the code paths gated behind `None` checks must be completely inert — byte-for-byte identical output.

**Primary recommendation:** Option A from architecture section — factory owns all dimension math and builds optional components; HashiGraphModel receives them as constructor args. This is already reflected in the existing PLAN.

## As-Built Component Verification

All Phase 4 components have been read directly. Contracts are confirmed:

### IterativeBackbone (src2/hashi_puzzle_solver/models/iterative_backbone.py)

Confirmed public attributes safe to call in the interleaved loop:
- `self.steps: int` — loop count
- `self.dropout: float` — for `F.dropout` call
- `self.conv: TransformerConv` — call as `self.conv(h, edge_index, edge_attr=edge_attr)`
- `self.norm: LayerNorm` — call as `self.norm(h)`
- `self.final_dim: int` — equals `hidden_channels` (output dim preserved)
- `concat=False` is unconditional — residual add needs no shape guard

Internal import: uses `torch.nn.functional as func` (alias is `func`, not `F`). The plan uses `F` — either alias works in `core.py` as long as the import is consistent with what is already there. `core.py` currently imports `torch` but NOT `torch.nn.functional`. Task 1 must add the import.

### ReverseBackbone (src2/hashi_puzzle_solver/models/reverse_backbone.py)

Confirmed:
- `self.project_embeddings: bool` — attribute accessible
- `self.final_dim: int` — set correctly for both `separate_weights=True/False`
- `self.projection: Linear(2 * final_dim, hidden_channels)` — only exists when `project_embeddings=True`
- `forward()` returns raw reverse embeddings `[N, final_dim]` — concatenation deferred to Phase 5
- `_get_backbone()` method used internally; external callers do not need it
- `object.__setattr__(self, "_shared_backbone", fwd)` used for shared weight case — NOT registered as submodule

**Critical**: `self.projection` is NOT called inside `ReverseBackbone.forward()`. The plan is correct that `HashiGraphModel.forward()` must call `self.reverse_backbone.projection(h_cat)` explicitly.

### HashiGraphModel (src2/hashi_puzzle_solver/models/core.py)

Current state — NOT yet updated for Phase 5:
- `__init__` has no `iterative_backbone` or `reverse_backbone` args
- Step 6 is exactly `h = self.backbone(h, edge_index, edge_attr=h_edge)`
- No `torch.nn.functional` import at module level (uses `torch.nn.Sequential`, `torch.nn.Linear`, `torch.nn.LayerNorm` directly)
- `noise_to_meta` linear uses `backbone.final_dim` — this is fine because noise meta operates on backbone output before composition (step 4 happens before step 6)

### ModelFactory (src2/hashi_puzzle_solver/models/factory.py)

Current state — NOT yet updated for Phase 5:
- EdgeHead line 52-56: `node_hidden_dim=backbone.final_dim` — must change to `edge_head_node_dim`
- ProphetHead line 60-63: `node_hidden_dim=backbone.final_dim` — must also change
- No imports of IterativeBackbone or ReverseBackbone

### HashiModelConfig.from_dict()

The `from_dict` classmethod handles `reasoning` and `reverse_gnn` nested dicts correctly. Integration tests can use either:
1. `HashiModelConfig.from_dict(dict)` — follows the dict pattern from `test_diffusion_rev_reason.py`
2. Direct dataclass construction: `HashiModelConfig(model=ModelConfig(reasoning=ReasoningConfig(enabled=True, steps=2), ...))`

Option 2 is cleaner for test fixtures since it avoids dict key coverage issues.

## Standard Stack

### Core (already in codebase, no new installs)

| Component | Location | Purpose |
|-----------|----------|---------|
| `IterativeBackbone` | `src2/hashi_puzzle_solver/models/iterative_backbone.py` | K-iteration shared-weight TransformerConv with residual |
| `ReverseBackbone` | `src2/hashi_puzzle_solver/models/reverse_backbone.py` | Backbone on reversed edges; holds `self.projection` |
| `HashiGraphModel` | `src2/hashi_puzzle_solver/models/core.py` | Shell to update |
| `EdgeHead` | `src2/hashi_puzzle_solver/models/heads.py` | Already parameterized by `node_hidden_dim`; just pass correct value |
| `ModelFactory` | `src2/hashi_puzzle_solver/models/factory.py` | Computes dims and assembles — needs dim calculation update |

### No New Libraries Required

All torch, torch_geometric, and linear algebra primitives are already imported across these files.

## Architecture Patterns

### Recommended Project Structure

No new files needed. Phase 5 modifies only:

```
src2/hashi_puzzle_solver/models/
├── core.py          MODIFY — add optional backbone args + composition block
└── factory.py       MODIFY — add optional component construction + dim math

tests/
└── test_hashi_graph_model_integration.py    CREATE — Wave 0 RED, then GREEN
```

### Pattern 1: Sequential (flags independent — no rev-reasoning)

When only one flag is enabled:

```python
# reasoning only:
h = self.backbone(h, edge_index, edge_attr=h_edge)
h = self.iterative_backbone(h, edge_index, edge_attr=h_edge)
# h shape: [N, hidden_channels]

# reverse only:
h = self.backbone(h, edge_index, edge_attr=h_edge)
h_rev = self.reverse_backbone(h, edge_index, edge_attr=h_edge)
h = torch.cat([h, h_rev], dim=-1)  # [N, 2 * final_dim]
if self.reverse_backbone.project_embeddings:
    h = self.reverse_backbone.projection(h)   # [N, hidden_channels]
```

### Pattern 2: Interleaved (both enabled — rev-reasoning)

Success criterion 3 requires that each reasoning iteration interleaves the reverse pass before the residual. The conv+norm+relu+dropout sequence from IterativeBackbone is replicated in HashiGraphModel.forward() directly, rather than calling `iterative_backbone.forward()`:

```python
# Source pattern: 05-integration-01-PLAN.md Task 1 action
h = self.backbone(h, edge_index, edge_attr=h_edge)  # forward backbone, always

if self.iterative_backbone is not None and self.reverse_backbone is not None:
    import torch.nn.functional as F  # use existing alias in core.py
    for _ in range(self.iterative_backbone.steps):
        h_in = h
        h_fwd = self.iterative_backbone.conv(h, edge_index, edge_attr=h_edge)
        h_fwd = self.iterative_backbone.norm(h_fwd)
        h_fwd = F.relu(h_fwd)
        h_fwd = F.dropout(h_fwd, p=self.iterative_backbone.dropout, training=self.training)
        h_rev = self.reverse_backbone(h, edge_index, edge_attr=h_edge)
        h_cat = torch.cat([h_fwd, h_rev], dim=-1)
        h = self.reverse_backbone.projection(h_cat)  # project_embeddings guaranteed True
        h = h + h_in
elif self.iterative_backbone is not None:
    h = self.iterative_backbone(h, edge_index, edge_attr=h_edge)
elif self.reverse_backbone is not None:
    h_rev = self.reverse_backbone(h, edge_index, edge_attr=h_edge)
    h = torch.cat([h, h_rev], dim=-1)
    if self.reverse_backbone.project_embeddings:
        h = self.reverse_backbone.projection(h)
# else: h passes unchanged — baseline path
```

**Note on `func` vs `F`**: `IterativeBackbone` uses `func` as its alias for `torch.nn.functional`. `core.py` does not currently import `torch.nn.functional` at all. Add `import torch.nn.functional as F` at the top of `core.py` before using it in the composition block.

### Pattern 3: Baseline (both disabled)

```python
h = self.backbone(h, edge_index, edge_attr=h_edge)
# h passes directly to EdgeHead — identical to pre-phase behavior
```

All new code is gated exclusively behind `if self.iterative_backbone is not None` and `if self.reverse_backbone is not None`. No config reading in `forward()`.

### EdgeHead Dimension Calculation

`EdgeHead.__init__` takes `node_hidden_dim` and computes `input_dim = 2 * node_hidden_dim` internally. The value passed must match the actual dimension of `h` at the point EdgeHead receives it:

| Flags | Post-composition dim | `node_hidden_dim` passed to EdgeHead |
|-------|---------------------|--------------------------------------|
| neither | `backbone.final_dim` | `backbone.final_dim` |
| reasoning only | `backbone.final_dim` (IterativeBackbone preserves dim) | `backbone.final_dim` |
| reverse only, `project_embeddings=True` | `hidden_channels` | `hidden_channels` |
| reverse only, `project_embeddings=False` | `2 * backbone.final_dim` | `2 * backbone.final_dim` |
| both, `project_embeddings=True` | `hidden_channels` | `hidden_channels` |

Factory computation before EdgeHead construction:

```python
# Source: 05-integration-01-PLAN.md Task 1 factory.py changes
edge_head_node_dim = backbone.final_dim
if model_config.reverse_gnn.enabled:
    if model_config.reverse_gnn.project_embeddings:
        edge_head_node_dim = model_config.hidden_channels
    else:
        edge_head_node_dim = 2 * backbone.final_dim
# reasoning.enabled alone does not change node embedding dim
```

### `__init__` Validation Constraint

When both are enabled, `project_embeddings=False` would leave `h_combined` at `2 * final_dim` while `h_in` is `hidden_channels` — residual add fails. Validate at construction time:

```python
# In HashiGraphModel.__init__, after storing the components:
if iterative_backbone is not None and reverse_backbone is not None:
    if not reverse_backbone.project_embeddings:
        raise ValueError(
            "When both reasoning and reverse_gnn are enabled, "
            "project_embeddings must be True (required for residual dimension match). "
            "Set model.reverse_gnn.project_embeddings: true in config."
        )
```

### Anti-Patterns to Avoid

- **Do not re-instantiate IterativeBackbone inside HashiGraphModel.forward()** — it must be registered as a submodule in `__init__` so its parameters are included in optimizer.
- **Do not call `iterative_backbone.forward()` in the interleaved path** — the interleaved loop replicates the internal steps manually to insert the reverse pass at the right place.
- **Do not pass `reverse_backbone` to `EdgeHead`** — EdgeHead receives a plain tensor.
- **Do not hardcode `node_hidden_dim` in the factory** — compute from `backbone.final_dim` and config flags.
- **Do not modify IterativeBackbone or ReverseBackbone** — Phase 5 wires them without changing them.
- **Do not use `object.__setattr__` for `iterative_backbone` or `reverse_backbone` in core.py** — use normal attribute assignment so they register as submodules.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Edge reversal | Custom flip logic | `edge_index.flip(0)` (already in ReverseBackbone) | Already implemented; just call `reverse_backbone(h, edge_index)` |
| Projection layer | Another Linear | `reverse_backbone.projection` (already registered) | ReverseBackbone already owns and registers this linear layer |
| Dimension inference | Conditional arithmetic at forward time | Static dim computation in factory at init | PyTorch modules need fixed shapes at init time for correct parameter registration |

## Common Pitfalls

### Pitfall 1: EdgeHead receives wrong node_hidden_dim

**What goes wrong:** `EdgeHead.__init__` computes `input_dim = 2 * node_hidden_dim`. If `node_hidden_dim` is `backbone.final_dim` but the actual `h` passed to the head has `2 * backbone.final_dim` (reverse concat without projection), the MLP input size is wrong and a shape mismatch error occurs at forward time.

**Why it happens:** The factory currently hardcodes `backbone.final_dim` for EdgeHead. This was correct before Phase 5.

**How to avoid:** Compute `edge_head_node_dim` dynamically from config flags in the factory before constructing EdgeHead. See EdgeHead Dimension Calculation above.

**Warning signs:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied` on the first forward pass in rev-reason mode.

### Pitfall 2: Forgetting to register optional components as submodules

**What goes wrong:** If `iterative_backbone` or `reverse_backbone` are stored via `object.__setattr__` rather than normal assignment, their parameters won't be in the optimizer.

**Why it happens:** Mistakenly reusing the `object.__setattr__` trick from the shared-weights case inside ReverseBackbone.

**How to avoid:** In `HashiGraphModel.__init__`, use normal attribute assignment: `self.iterative_backbone = iterative_backbone`. PyTorch's `nn.Module.__setattr__` registers it as a submodule when the value is an `nn.Module`.

**Warning signs:** `len(list(model.parameters()))` smaller than expected; new components' weights don't appear in optimizer state dict.

### Pitfall 3: Residual dimension mismatch in interleaved mode

**What goes wrong:** In the interleaved loop (both enabled), after concatenating `[h_fwd, h_rev]` and projecting, if `project_embeddings=False` was somehow allowed through, `h_combined` would be `2 * final_dim` while `h_in` is `hidden_channels`.

**Why it happens:** `project_embeddings=False` with both flags enabled leaves `h_combined` at wrong dimension.

**How to avoid:** Validate at `HashiGraphModel.__init__` time with a `ValueError` as described above. This prevents the forward call from ever reaching the broken state.

**Warning signs:** Shape mismatch on `h = h_combined + h_in` if the validation is skipped.

### Pitfall 4: Baseline non-equivalence

**What goes wrong:** When both flags are disabled, the output differs from the pre-phase baseline.

**Why it happens:** Any accidental activation of a new code path — e.g., running `iterative_backbone` even when it's `None` due to missing guard, or changing the order of operations.

**How to avoid:** Gate ALL new code behind `if self.iterative_backbone is not None:` and `if self.reverse_backbone is not None:`. No config-reading in `forward()`.

**Warning signs:** Test asserting byte-for-byte equivalence fails when both flags are disabled.

### Pitfall 5: IterativeBackbone edge_dim mismatch

**What goes wrong:** `IterativeBackbone` is constructed with `edge_dim=backbone_edge_dim` but at forward time receives `h_edge` of a different dimension.

**Why it happens:** `h_edge` in `HashiGraphModel.forward()` may include the noise projection concatenation (step 5). The `IterativeBackbone` is called with the same `h_edge` as the backbone.

**How to avoid:** Pass `backbone_edge_dim` (which already accounts for noise) to `IterativeBackbone.__init__` in the factory. Both backbone and iterative_backbone receive the same `h_edge`.

**Warning signs:** `TransformerConv` raises dimension error on first forward pass with noise enabled.

### Pitfall 6: Missing `torch.nn.functional` import in core.py

**What goes wrong:** The interleaved loop calls `F.relu(h_fwd)` and `F.dropout(...)` but `core.py` does not currently import `torch.nn.functional`.

**Why it happens:** The existing `core.py` imports `torch`, `torch.nn.Linear`, `torch.nn.ReLU`, `torch.nn.Sequential` directly — it never needed `F`.

**How to avoid:** Add `import torch.nn.functional as F` at the top of `core.py` in Task 1.

## Code Examples

### How the forward method currently ends (step 6-7)

```python
# Source: src2/hashi_puzzle_solver/models/core.py lines 120-131
# 6. Message Passing (Backbone)
h = self.backbone(h, edge_index, edge_attr=h_edge)

# 7. Prediction Heads
edge_logits = self.edge_head(
    h,
    edge_index,
    edge_attr=h_edge,
    node_type=node_type,
    batch=batch,
    noise_emb=noise_emb,
)
```

### EdgeHead constructor signature (unchanged — interface only)

```python
# Source: src2/hashi_puzzle_solver/models/heads.py
class EdgeHead(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        node_hidden_dim: int,   # <-- this is what changes in factory
        edge_attr_dim: int,
    ):
        input_dim = 2 * node_hidden_dim  # base: src + dst
        # + additional dims for meta, noise, etc.
```

### Factory current EdgeHead construction (to be updated)

```python
# Source: src2/hashi_puzzle_solver/models/factory.py lines 52-56
edge_head = EdgeHead(
    model_config,
    node_hidden_dim=backbone.final_dim,   # <-- must become edge_head_node_dim
    edge_attr_dim=backbone_edge_dim
)
```

### ReverseBackbone.projection usage (how to call it in core.py)

```python
# Source: src2/hashi_puzzle_solver/models/reverse_backbone.py lines 68-70
# self.projection = Linear(2 * self.final_dim, hidden_channels)
# Call in HashiGraphModel.forward():
h_cat = torch.cat([h_fwd, h_rev], dim=-1)    # [N, 2 * final_dim]
h = self.reverse_backbone.projection(h_cat)   # [N, hidden_channels]
```

### IterativeBackbone internals for interleaved access

```python
# Source: src2/hashi_puzzle_solver/models/iterative_backbone.py
# These attributes are public and safe to call in HashiGraphModel:
# self.iterative_backbone.conv   — TransformerConv (call: conv(h, edge_index, edge_attr=...))
# self.iterative_backbone.norm   — LayerNorm (call: norm(h))
# self.iterative_backbone.dropout — float (NOT a module; use F.dropout(h, p=..., training=self.training))
# self.iterative_backbone.steps   — int
```

### Config construction for integration tests

```python
# Direct dataclass construction (cleaner for test fixtures):
from hashi_puzzle_solver.models.config import (
    HashiModelConfig, ModelConfig, ReasoningConfig, ReverseGnnConfig
)
config = HashiModelConfig(
    model=ModelConfig(
        type="transformer",
        hidden_channels=16,
        num_layers=2,
        heads=1,
        dropout=0.0,
        use_noise_head=False,
        use_noise_in_message_passing=False,
        use_noise_in_prediction=False,
        use_noise_in_global_meta=False,
        reasoning=ReasoningConfig(enabled=True, steps=2),
        reverse_gnn=ReverseGnnConfig(enabled=True, project_embeddings=True),
    )
)
```

### Test spy pattern for counting reverse_backbone calls

```python
# Same pattern used in test_iterative_backbone.py (conv.forward spy):
call_count = 0
_original_forward = model.reverse_backbone.forward

def spy_forward(*args, **kwargs):
    nonlocal call_count
    call_count += 1
    return _original_forward(*args, **kwargs)

model.reverse_backbone.forward = spy_forward  # type: ignore[method-assign]
# Run model.forward(...)
assert call_count == expected_steps
```

## State of the Art

| Old Pattern | Phase 5 Pattern | Notes |
|-------------|----------------|-------|
| `h = backbone(h, ...)` single call | conditional composition block (3 branches + pass-through) | gated entirely by None checks |
| `EdgeHead(node_hidden_dim=backbone.final_dim)` | `EdgeHead(node_hidden_dim=edge_head_node_dim)` | computed from flags in factory |
| `HashiGraphModel` has no optional GNN components | optional `iterative_backbone`, `reverse_backbone` submodules | registered via normal `self.x = x` assignment |

## Open Questions

1. **ProphetHead node_hidden_dim update**
   - What we know: `ProphetHead` is constructed with `node_hidden_dim=backbone.final_dim` in factory.py. Its input path uses `h[meta_mask]` which after Phase 5 may be a different dimension.
   - What's unclear: Whether ProphetHead is ever active in `rev-reason` mode (it is a noise prediction head, diffusion-specific). In rev-reason, `use_noise_head` is likely False.
   - Recommendation: Update `ProphetHead` construction in factory to use `edge_head_node_dim` for consistency. If `use_noise_head=False` in rev-reason, this is a no-op in practice but keeps the factory correct.

2. **edge_attr directionality for ReverseBackbone**
   - What we know: `ReverseBackbone.forward()` passes `edge_attr` as-is (not flipped). The class docstring notes "Phase 5 integration can address this."
   - What's unclear: Whether edge attributes are directional.
   - Recommendation: Pass `h_edge` as-is to `reverse_backbone` for Phase 5 (simplest, consistent with current Phase 4 implementation). Directional edge attributes can be a separate future improvement.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` with `testpaths = ["tests"]`, `pythonpath = ["src2"]` |
| Quick run command | `pytest tests/test_hashi_graph_model_integration.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements to Test Map

Phase 5 carries no new requirement IDs. Its success criteria map directly to test behaviors:

| Success Criterion | Behavior | Test Type | Automated Command | File Exists? |
|-------------------|----------|-----------|-------------------|-------------|
| SC-1: reasoning K iters + reverse concat + projection | Forward pass with both enabled produces correct-shape output | unit | `pytest tests/test_hashi_graph_model_integration.py::test_both_flags_enabled -x -q` | No — Wave 0 |
| SC-2: all flags disabled = byte-for-byte identical | Two forward passes with disabled flags produce equal output | unit | `pytest tests/test_hashi_graph_model_integration.py::test_flags_disabled_baseline -x -q` | No — Wave 0 |
| SC-3: interleaved rev-reasoning | spy on reverse_backbone.forward; assert called exactly K times | unit | `pytest tests/test_hashi_graph_model_integration.py::test_rev_reasoning_interleaved -x -q` | No — Wave 0 |
| SC-4: EdgeHead correct dim in all combos | 5 flag combos complete without shape RuntimeError | unit | `pytest tests/test_hashi_graph_model_integration.py::test_edge_head_dim_all_combos -x -q` | No — Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/test_hashi_graph_model_integration.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_hashi_graph_model_integration.py` — covers all 4 success criteria:
  - `test_flags_disabled_baseline` — byte-for-byte equivalence (SC-2)
  - `test_reasoning_only` — IterativeBackbone wired, output shape correct (SC-1)
  - `test_reverse_only_with_projection` — ReverseBackbone concat + projection, correct EdgeHead dim (SC-4)
  - `test_reverse_only_no_projection` — `2 * final_dim` passed to EdgeHead, no shape error (SC-4)
  - `test_both_flags_enabled` — rev-reasoning composition, shapes correct (SC-1)
  - `test_rev_reasoning_interleaved` — K iterations each using fwd + rev + residual; spy counts K calls (SC-3)
  - `test_edge_head_dim_all_combos` — 5 flag combinations produce no shape mismatch (SC-4)

*(All other test infrastructure in `tests/` and `tests_src2/` covers prior phases — no framework install needed. Pre-existing failure in `tests/test_adaptive_sampler.py` due to `ModuleNotFoundError: No module named 'hashi_puzzle_solver.diffusion_engine'` is unrelated to Phase 5 and should be ignored.)*

## Sources

### Primary (HIGH confidence)

- Direct codebase read: `src2/hashi_puzzle_solver/models/iterative_backbone.py` — as-built component, confirmed all public attributes
- Direct codebase read: `src2/hashi_puzzle_solver/models/reverse_backbone.py` — as-built component, confirmed projection/forward contract
- Direct codebase read: `src2/hashi_puzzle_solver/models/core.py` — confirmed current state (no Phase 5 changes yet)
- Direct codebase read: `src2/hashi_puzzle_solver/models/factory.py` — confirmed current state (no Phase 5 changes yet)
- Direct codebase read: `src2/hashi_puzzle_solver/models/config.py` — confirmed ReasoningConfig, ReverseGnnConfig, HashiModelConfig.from_dict()
- Direct codebase read: `tests/test_iterative_backbone.py`, `tests/test_reverse_backbone.py` — confirmed spy pattern works; established fixture patterns
- Direct codebase read: `tests/test_diffusion_rev_reason.py` — confirmed config dict shape for test fixtures
- `.planning/phases/04-component-implementation/04-component-implementation-02-SUMMARY.md` — confirmed Phase 4 plan 02 delivery
- `.planning/phases/04-component-implementation/04-component-implementation-03-SUMMARY.md` — confirmed Phase 4 plan 03 delivery

### Secondary (MEDIUM confidence)

- ROADMAP.md Phase 5 success criteria — used to derive composition requirements
- STATE.md `## Decisions` — design decisions locked in previous phases

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all components exist and were read directly
- Architecture (sequential/reverse-only paths): HIGH — dimension math is deterministic from code
- Architecture (interleaved both-enabled path): HIGH — success criterion 3 confirmed to require interleaving; PLAN already specifies the exact loop
- Pitfalls: HIGH — derived from direct code inspection including new pitfall 6 (missing F import)
- EdgeHead dimension: HIGH — heads.py constructor confirmed, factory line identified

**Research date:** 2026-03-09 (updated 2026-03-09)
**Valid until:** 2026-04-09 (stable codebase; only changes if Phase 5 introduces new components)
