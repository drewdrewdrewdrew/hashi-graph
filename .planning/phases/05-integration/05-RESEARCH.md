# Phase 5: Integration - Research

**Researched:** 2026-03-09
**Domain:** PyTorch nn.Module composition — wiring IterativeBackbone and ReverseBackbone into HashiGraphModel.forward(), EdgeHead dimension adaptation
**Confidence:** HIGH

## Summary

Phase 5 is a pure in-codebase integration task. No new libraries are needed. All components are already implemented and independently tested. The work is:

1. **HashiGraphModel.__init__**: conditionally construct `IterativeBackbone` and/or `ReverseBackbone` from config flags; compute the correct `node_hidden_dim` to pass to `EdgeHead`.
2. **HashiGraphModel.forward()**: replace the current single `self.backbone(h, ...)` call with a composition block that runs `IterativeBackbone` K times (if enabled) and/or `ReverseBackbone` in parallel (if enabled), then concatenates and projects as needed.
3. **ModelFactory.create_model**: pass the `IterativeBackbone` and `ReverseBackbone` instances into `HashiGraphModel`, or update the construction logic so the model builds them internally from config.
4. **Backward-compatibility guard**: with all flags disabled the output must be byte-for-byte identical to the pre-phase baseline.

The key design question is *where* the new components are constructed: inside `HashiGraphModel.__init__` (self-contained) or in `ModelFactory`. Based on how `backbone` is already injected via the factory, the cleanest approach is to build the optional components inside `HashiGraphModel.__init__` directly from `config.model.reasoning` and `config.model.reverse_gnn`, keeping the factory unchanged.

**Primary recommendation:** Build `IterativeBackbone` and `ReverseBackbone` inside `HashiGraphModel.__init__` from `config.model.reasoning`/`config.model.reverse_gnn`; update `forward()` to compose them; compute `EdgeHead` `node_hidden_dim` after determining post-composition embedding size.

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

### Recommended Composition Logic in `HashiGraphModel.forward()`

The composition replaces step 6 (backbone call) in the existing `forward()`:

```
Current step 6:
    h = self.backbone(h, edge_index, edge_attr=h_edge)

Phase 5 step 6 (new):
    h = self.backbone(h, edge_index, edge_attr=h_edge)        # always: forward backbone

    if self.iterative_backbone is not None:
        h = self.iterative_backbone(h, edge_index, edge_attr=h_edge)   # K reasoning iterations

    if self.reverse_backbone is not None:
        h_rev = self.reverse_backbone(h, edge_index, edge_attr=h_edge) # reverse pass
        h = torch.cat([h, h_rev], dim=-1)                              # [N, 2 * final_dim]
        if self.reverse_backbone.project_embeddings:
            h = self.reverse_backbone.projection(h)                    # [N, hidden_channels]
```

This ordering satisfies success criterion 3: when both are enabled, reasoning iterates over the forward backbone first, *then* the reverse backbone augments the final embedding before the EdgeHead.

**Important**: Success criterion 3 says "each reasoning iteration uses forward + reverse passes before the residual update". This is a stricter requirement than a simple sequential composition. Re-read carefully:

> "With both enabled (rev-reasoning), each reasoning iteration uses forward + reverse passes before the residual update"

This means the interleave pattern should be: for each reasoning step, run forward conv + reverse pass + combine, then apply residual. This is *different* from: run all forward reasoning first, then run reverse.

There are two valid interpretations:

**Interpretation A (simpler, likely intended):** "rev-reasoning" means reasoning (iterative forward) AND reverse GNN are both active — the two components coexist but their passes do not interleave per-iteration. IterativeBackbone runs K iterations internally; ReverseBackbone runs once in parallel. This is the straightforward sequential composition above.

**Interpretation B (strict literal):** Within each iteration of IterativeBackbone, a reverse pass is inserted before the residual. This would require `IterativeBackbone` to accept an optional `reverse_backbone` argument.

Given that Phase 4 kept `IterativeBackbone` and `ReverseBackbone` completely standalone, and Phase 5's success criteria say "each reasoning iteration uses forward + reverse passes before the residual update", **Interpretation B is the intended design**. The planner must implement an interleaved loop.

### Pattern 1: Sequential (flags independent — no rev-reasoning)

When only one flag is enabled, the components are fully independent:

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

When both flags are enabled, each iteration of the reasoning loop also incorporates the reverse pass before the residual:

```python
h = self.backbone(h, edge_index, edge_attr=h_edge)  # forward backbone, always

if both_enabled:
    for _ in range(self.iterative_backbone.steps):
        h_in = h
        h_fwd = self.iterative_backbone.conv(h, edge_index, edge_attr=h_edge)
        h_fwd = self.iterative_backbone.norm(h_fwd)
        h_fwd = F.relu(h_fwd)
        h_fwd = F.dropout(h_fwd, p=self.iterative_backbone.dropout, training=self.training)

        h_rev = self.reverse_backbone(h, edge_index, edge_attr=h_edge)  # [N, final_dim]
        # combine fwd + rev before residual
        h_combined = torch.cat([h_fwd, h_rev], dim=-1)  # [N, 2 * final_dim]
        if self.reverse_backbone.project_embeddings:
            h_combined = self.reverse_backbone.projection(h_combined)  # [N, hidden_channels]
        # residual: h_combined must be same dim as h_in
        h = h_combined + h_in
```

**Critical dimension constraint**: For the residual `h = h_combined + h_in` to work, `h_combined` must have the same dimension as `h_in` (= `hidden_channels`). This requires `project_embeddings=True` in the interleaved case, which is the default. If `project_embeddings=False` with both enabled, the residual would fail (dim mismatch). The implementation should either:
- Raise an error if both enabled and `project_embeddings=False`
- Or skip the residual in that case (but this breaks IterativeBackbone's contract)

The cleanest approach: when both are enabled, always apply the projection (even if `project_embeddings=False` in config), OR enforce that `project_embeddings=True` is required when both flags are on.

**Recommendation**: In `HashiGraphModel.__init__`, if `reasoning.enabled and reverse_gnn.enabled`, assert/error if `project_embeddings=False`. Document this constraint clearly.

### Pattern 3: Baseline (both disabled)

```python
h = self.backbone(h, edge_index, edge_attr=h_edge)
# h passes directly to EdgeHead — no change from current behavior
```

This must be byte-for-byte identical to pre-phase output.

### EdgeHead Dimension Calculation

`EdgeHead.__init__` takes `node_hidden_dim` as the node embedding size seen by the head. This value depends on which flags are active:

| Flags | Post-composition dim | `node_hidden_dim` passed to EdgeHead |
|-------|---------------------|--------------------------------------|
| neither | `backbone.final_dim` | `backbone.final_dim` |
| reasoning only | `backbone.final_dim` (IterativeBackbone preserves dim) | `backbone.final_dim` |
| reverse only, project_embeddings=True | `hidden_channels` | `hidden_channels` |
| reverse only, project_embeddings=False | `2 * backbone.final_dim` | `2 * backbone.final_dim` |
| both, project_embeddings=True | `hidden_channels` | `hidden_channels` |

Since `EdgeHead` is constructed once at init time, the `node_hidden_dim` must be computed statically before calling the constructor. `ModelFactory.create_model` currently passes `backbone.final_dim` — this needs to change.

**Key insight**: The factory needs to compute `edge_head_node_dim` based on config flags:

```python
edge_head_node_dim = backbone.final_dim
if model_config.reverse_gnn.enabled:
    if model_config.reverse_gnn.project_embeddings:
        edge_head_node_dim = model_config.hidden_channels
    else:
        edge_head_node_dim = 2 * backbone.final_dim
# reasoning.enabled alone does not change node_hidden_dim
```

OR, since `HashiGraphModel.__init__` will own the optional components, it can expose a `node_output_dim` property that the factory reads. The simpler approach: compute the dim in the factory before constructing `EdgeHead`, then pass both the computed dim AND the new components into `HashiGraphModel`.

### Where to Construct Optional Components

Two options:

**Option A: In ModelFactory.create_model**
- Factory builds `IterativeBackbone`, `ReverseBackbone` if enabled
- Passes them as constructor args to `HashiGraphModel`
- `HashiGraphModel.__init__` receives them as optional `nn.Module | None` args
- Factory computes the correct `node_hidden_dim` for `EdgeHead`

**Option B: In HashiGraphModel.__init__**
- `HashiGraphModel.__init__` reads `config.model.reasoning` and builds internally
- Factory does not need to know about the new components
- But factory still needs to pass the correct `node_hidden_dim` to `EdgeHead`... which it can't compute without knowing what the model will build

**Recommendation: Option A** — factory owns all dimension math. `HashiGraphModel.__init__` receives optional `iterative_backbone` and `reverse_backbone` args. This keeps dimension logic in one place and makes the model shell testable with injected components.

Factory change:

```python
from .iterative_backbone import IterativeBackbone
from .reverse_backbone import ReverseBackbone

# After building backbone:
iterative_bb = None
if model_config.reasoning.enabled:
    iterative_bb = IterativeBackbone(
        hidden_channels=model_config.hidden_channels,
        steps=model_config.reasoning.steps,
        heads=model_config.heads,
        dropout=model_config.dropout,
        edge_dim=backbone_edge_dim,
    )

reverse_bb = None
if model_config.reverse_gnn.enabled:
    reverse_bb = ReverseBackbone(
        forward_backbone=backbone,
        hidden_channels=model_config.hidden_channels,
        separate_weights=model_config.reverse_gnn.separate_weights,
        project_embeddings=model_config.reverse_gnn.project_embeddings,
    )

# Compute correct dim for EdgeHead
edge_head_node_dim = backbone.final_dim
if model_config.reverse_gnn.enabled:
    if model_config.reverse_gnn.project_embeddings:
        edge_head_node_dim = model_config.hidden_channels
    else:
        edge_head_node_dim = 2 * backbone.final_dim

edge_head = EdgeHead(model_config, node_hidden_dim=edge_head_node_dim, edge_attr_dim=backbone_edge_dim)

model = HashiGraphModel(
    config=config,
    ...
    iterative_backbone=iterative_bb,
    reverse_backbone=reverse_bb,
)
```

### Anti-Patterns to Avoid

- **Do not re-instantiate IterativeBackbone inside HashiGraphModel.forward()** — it must be registered as a submodule in `__init__` so its parameters are included in optimizer.
- **Do not pass `reverse_backbone` to `EdgeHead`** — EdgeHead should receive a plain tensor; let `HashiGraphModel.forward()` handle all composition before calling the head.
- **Do not hardcode `node_hidden_dim` in the factory** — compute it from `backbone.final_dim` and the config flags.
- **Do not modify IterativeBackbone or ReverseBackbone** — Phase 5 wires them; it does not change them.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Edge reversal | Custom flip logic | `edge_index.flip(0)` (already in ReverseBackbone) | Already implemented; just call `reverse_backbone(h, edge_index)` |
| Projection layer | Another Linear | `reverse_backbone.projection` (already registered) | ReverseBackbone already owns and registers this linear layer |
| Dimension inference | Conditional arithmetic at forward time | Static dim computation in factory at init | PyTorch modules need fixed shapes at init time for correct parameter registration |

## Common Pitfalls

### Pitfall 1: EdgeHead receives wrong node_hidden_dim

**What goes wrong:** `EdgeHead.__init__` computes `input_dim = 2 * node_hidden_dim`. If `node_hidden_dim` is `backbone.final_dim` but the actual `h` passed to the head has `2 * backbone.final_dim` (reverse concat without projection), the MLP input size is wrong and a shape mismatch error occurs at forward time.

**Why it happens:** The factory currently hardcodes `backbone.final_dim` for `EdgeHead`. This was correct before Phase 5.

**How to avoid:** Compute `edge_head_node_dim` dynamically from config flags in the factory before constructing `EdgeHead`. See "EdgeHead Dimension Calculation" above.

**Warning signs:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied` on the first forward pass in rev-reason mode.

### Pitfall 2: Forgetting to register optional components as submodules

**What goes wrong:** If `iterative_backbone` or `reverse_backbone` are stored via `object.__setattr__` (like the shared-weights pattern in `ReverseBackbone`) rather than normal assignment, their parameters won't be in the optimizer. The model trains but the new components don't learn.

**Why it happens:** Mistakenly reusing the `object.__setattr__` trick from the shared-weights case.

**How to avoid:** In `HashiGraphModel.__init__`, use normal attribute assignment: `self.iterative_backbone = iterative_backbone`. PyTorch's `nn.Module.__setattr__` will register it correctly as a submodule when the value is not None.

**Warning signs:** `len(list(model.parameters()))` is smaller than expected; new components' weights don't appear in optimizer state dict.

### Pitfall 3: Residual dimension mismatch in interleaved mode

**What goes wrong:** In the interleaved loop (both enabled), after concatenating `[h_fwd, h_rev]` and optionally projecting, the result may not be `hidden_channels` if `project_embeddings=False`. Residual add `h = h_combined + h_in` then fails.

**Why it happens:** `project_embeddings=False` with both flags enabled leaves `h_combined` at `2 * final_dim` while `h_in` is `hidden_channels`.

**How to avoid:** Either assert `project_embeddings=True` when both flags are enabled, or unconditionally apply the projection in the interleaved case (creation of a separate projection in `HashiGraphModel.__init__` if `project_embeddings=False` is allowed).

**Warning signs:** Shape mismatch on `h = h_combined + h_in` with both flags enabled and `project_embeddings: false`.

### Pitfall 4: Baseline non-equivalence

**What goes wrong:** When both flags are disabled, the output differs from the pre-phase baseline.

**Why it happens:** Any change to the backbone call path, such as accidentally activating a new code path, running `iterative_backbone` even when it's `None`, or changing the order of operations.

**How to avoid:** Gate ALL new code behind `if self.iterative_backbone is not None:` and `if self.reverse_backbone is not None:`. The `None` check must be the only guard — no config-reading in `forward()`.

**Warning signs:** Test asserting byte-for-byte equivalence fails when both flags are disabled.

### Pitfall 5: IterativeBackbone edge_dim mismatch

**What goes wrong:** `IterativeBackbone` is constructed with `edge_dim=backbone_edge_dim` but at forward time it receives `h_edge` of a different dimension (e.g., edge_dim includes noise embedding or does not).

**Why it happens:** `h_edge` in `HashiGraphModel.forward()` may include the noise projection concatenation (step 5). The `IterativeBackbone` is called with the same `h_edge` as the backbone.

**How to avoid:** Pass `backbone_edge_dim` (which already accounts for noise) to `IterativeBackbone.__init__` in the factory. Both `backbone` and `iterative_backbone` receive the same `h_edge`, so using `backbone_edge_dim` is correct.

## Code Examples

### How the forward method currently ends (step 6-7)

```python
# Source: src2/hashi_puzzle_solver/models/core.py lines 121-131
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

### EdgeHead constructor signature (already accepts node_hidden_dim)

```python
# Source: src2/hashi_puzzle_solver/models/heads.py lines 18-52
class EdgeHead(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        node_hidden_dim: int,   # <-- this is what changes
        edge_attr_dim: int,
    ):
        input_dim = 2 * node_hidden_dim  # base: src + dst
        # + additional dims for meta, noise, etc.
```

### Factory current EdgeHead construction

```python
# Source: src2/hashi_puzzle_solver/models/factory.py lines 52-56
edge_head = EdgeHead(
    model_config,
    node_hidden_dim=backbone.final_dim,   # <-- must become edge_head_node_dim
    edge_attr_dim=backbone_edge_dim
)
```

### ReverseBackbone.projection usage (how to call it)

```python
# Source: src2/hashi_puzzle_solver/models/reverse_backbone.py lines 68-70
# self.projection = Linear(2 * self.final_dim, hidden_channels)
# Call in HashiGraphModel.forward():
h_cat = torch.cat([h_fwd, h_rev], dim=-1)   # [N, 2 * final_dim]
h = self.reverse_backbone.projection(h_cat)  # [N, hidden_channels]
```

### IterativeBackbone internals for interleaved access

```python
# Source: src2/hashi_puzzle_solver/models/iterative_backbone.py lines 50-63
# These attributes are public and safe to call in HashiGraphModel:
# self.iterative_backbone.conv   — the TransformerConv
# self.iterative_backbone.norm   — the LayerNorm
# self.iterative_backbone.dropout — float
# self.iterative_backbone.steps   — int
```

## State of the Art

| Old Pattern | Phase 5 Pattern | Notes |
|-------------|----------------|-------|
| `h = backbone(h, ...)` single call | conditional composition block | gated by None checks |
| `EdgeHead(node_hidden_dim=backbone.final_dim)` | `EdgeHead(node_hidden_dim=edge_head_node_dim)` | computed from flags |
| `HashiGraphModel` has no optional GNN components | optional `iterative_backbone`, `reverse_backbone` submodules | registered via normal `self.x = x` assignment |

## Open Questions

1. **Interleaved vs. sequential when both enabled**
   - What we know: Success criterion 3 says "each reasoning iteration uses forward + reverse passes before the residual update" — this implies interleaving per-iteration.
   - What's unclear: Whether this means the `ReverseBackbone` is called inside each IterativeBackbone step, or whether "forward + reverse" means the two backbones both run (sequentially) and together constitute one "pass" per iteration.
   - Recommendation: Implement the interleaved inner loop (Pattern 2 above). If performance concerns arise, sequential composition can be a fallback.

2. **ProphetHead node_hidden_dim**
   - What we know: `ProphetHead` is constructed with `node_hidden_dim=backbone.final_dim`. Its input is `[meta_embedding, pooled_stats]` where `meta_embedding` comes from `h[meta_mask]`.
   - What's unclear: After Phase 5, `h` may have a different dimension (e.g., `hidden_channels` after projection). `ProphetHead.__init__` uses `node_hidden_dim + stats_dim` as input.
   - Recommendation: Update `ProphetHead` construction in the factory to use the same `edge_head_node_dim`. If `ProphetHead` is never active in `rev-reason` mode (it's noise prediction, diffusion-specific), this may be a non-issue in practice, but the constructor should be correct.

3. **edge_attr directionality for ReverseBackbone**
   - What we know: `ReverseBackbone.forward()` passes `edge_attr` as-is (not flipped). The note in `reverse_backbone.py` says "Phase 5 integration can address this."
   - What's unclear: Whether edge attributes in this graph are directional (e.g., edge type differs by direction).
   - Recommendation: Keep passing `h_edge` as-is to `reverse_backbone` for Phase 5 (simplest, aligns with current implementation). Directional edge attributes can be a separate future improvement.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | none (auto-discovered) |
| Quick run command | `pytest tests/test_hashi_graph_model_integration.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements to Test Map

Phase 5 carries no new requirement IDs. Its success criteria map directly to test behaviors:

| Success Criterion | Behavior | Test Type | Automated Command | File Exists? |
|-------------------|----------|-----------|-------------------|-------------|
| SC-1: reasoning K iters + reverse concat + projection | Forward pass with both enabled produces output; shapes correct | unit | `pytest tests/test_hashi_graph_model_integration.py::test_both_flags_enabled -x -q` | No — Wave 0 |
| SC-2: all flags disabled = byte-for-byte identical | Output with disabled flags equals pre-phase baseline | unit | `pytest tests/test_hashi_graph_model_integration.py::test_flags_disabled_baseline -x -q` | No — Wave 0 |
| SC-3: interleaved rev-reasoning | K iterations each using fwd + rev + residual | unit | `pytest tests/test_hashi_graph_model_integration.py::test_rev_reasoning_interleaved -x -q` | No — Wave 0 |
| SC-4: EdgeHead correct dim in all combinations | 4 flag combos × project_embeddings variations complete without shape error | unit | `pytest tests/test_hashi_graph_model_integration.py::test_edge_head_dim_all_combos -x -q` | No — Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/test_hashi_graph_model_integration.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_hashi_graph_model_integration.py` — covers all 4 success criteria
  - `test_flags_disabled_baseline` — byte-for-byte equivalence
  - `test_reasoning_only` — IterativeBackbone wired, output shape correct
  - `test_reverse_only_with_projection` — ReverseBackbone concat + projection, correct EdgeHead dim
  - `test_reverse_only_no_projection` — `2 * final_dim` passed to EdgeHead, no shape error
  - `test_both_flags_enabled` — rev-reasoning composition, shapes correct
  - `test_edge_head_dim_all_combos` — 4 flag combinations produce no shape mismatch

## Sources

### Primary (HIGH confidence)

- Direct codebase inspection: `src2/hashi_puzzle_solver/models/core.py`, `heads.py`, `backbone.py`, `iterative_backbone.py`, `reverse_backbone.py`, `factory.py`, `config.py` — all read and analyzed
- Phase 4 plans (04-02-PLAN.md, 04-03-PLAN.md) — confirmed component contracts
- Existing tests `tests/test_iterative_backbone.py`, `tests/test_reverse_backbone.py` — confirmed what Phase 4 delivered

### Secondary (MEDIUM confidence)

- ROADMAP.md Phase 5 success criteria — used to derive composition requirements
- STATE.md `## Decisions` — design decisions locked in previous phases (object.__setattr__ pattern, concat=False on IterativeBackbone, projection in ReverseBackbone)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all components exist and are read directly
- Architecture (sequential/reverse-only paths): HIGH — dimension math is deterministic
- Architecture (interleaved both-enabled path): MEDIUM — success criterion 3 requires interleaving but the exact loop structure needs planner confirmation
- Pitfalls: HIGH — derived from direct code inspection
- EdgeHead dimension: HIGH — heads.py constructor read directly

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (stable codebase; only changes if Phase 5 introduces new components)
