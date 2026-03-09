# Phase 4: Component Implementation - Research

**Researched:** 2026-03-09
**Domain:** PyTorch GNN components — iterative shared-weight TransformerConv, reverse backbone, trainer dispatch
**Confidence:** HIGH (all findings drawn directly from reading the current codebase)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODE-01 | `training.mode = "rev-reason"` routes to a new training path in `DiffusionTrainer.run_epoch` with no noise injection on edges | Trainer dispatch pattern is already established for `diff-cont`, `flow-blind`, `diff-discrete`. The `else` branch in `run_epoch` (line ~381) calls `inject_noise`; `rev-reason` needs its own `elif` that skips all noise injection and sets `data = batch` directly. |
| MODE-02 | Within `rev-reason`, `reasoning.enabled` and `reverse_gnn.enabled` independently activate their components (either, both, or neither) | `ModelConfig` already has typed `reasoning` and `reverse_gnn` fields. The dispatch path (04-01) passes `data` straight to the forward pass; component activation happens inside `HashiGraphModel` (Phase 5). For Phase 4, the dispatch path only needs to verify both components are independently check-able via config flags. |
| REAS-01 | When `reasoning.enabled: true`, a single shared-weight TransformerConv layer is applied K times with residual updates | `GraphBackbone` shows the exact TransformerConv constructor pattern. A single `TransformerConv` instance stored in `self.conv` is called K times in `forward()`. Residual pattern already used in `GraphBackbone.forward()` (lines 83-98). |
| REAS-02 | Number of iterations controlled by `reasoning.steps` | `steps` is available as `config.reasoning.steps` (int, validated >= 1). Loop `for _ in range(self.steps)` in `forward()`. |
| REVG-01 | When `reverse_gnn.enabled: true`, a reverse backbone runs on same input, output concatenated with forward embeddings | `GraphBackbone` can be reused/wrapped. Reverse backbone accepts same `(h, edge_index, edge_attr)` as forward. Output shape must match forward backbone's `final_dim`. |
| REVG-02 | `separate_weights=True` gives independent parameters; `separate_weights=False` shares forward backbone weights | When `separate_weights=False`, `ReverseBackbone` holds a reference to the external `GraphBackbone` instance and calls it directly. When `True`, it creates and owns its own `GraphBackbone`. |
| REVG-03 | `project_embeddings=True` adds a linear layer compressing concatenated embeddings to `hidden_channels` | Linear projection: `nn.Linear(2 * backbone.final_dim, hidden_channels)`. Applied inside `ReverseBackbone.forward()` when `self.project_embeddings is True`. |
</phase_requirements>

---

## Summary

Phase 4 splits into three parallel plans writing to three non-conflicting files. The work is grounded in patterns already established in the codebase — there is nothing speculative to discover from external sources.

**Plan 04-01** adds a `rev-reason` branch to `DiffusionTrainer.run_epoch` in `trainers/diffusion.py`. The existing `if mode == "diff-cont": ... elif mode == "flow-blind": ... else: inject_noise(...)` chain needs a new `elif mode == "rev-reason": data = batch` arm that bypasses all noise functions. The existing forward-pass and loss computation below the mode branch can be reused as-is for Phase 4 (component wiring happens in Phase 5).

**Plan 04-02** creates a new `IterativeBackbone` class in a new file under `models/`. It wraps a single `TransformerConv` (same constructor as the ones in `GraphBackbone`) applied K times with residual updates. The only nuance is dimension handling: unlike `GraphBackbone`'s multi-layer stack where `curr_dim` varies per layer, a shared-weight iterative layer must fix `in_channels == out_channels`. Forcing `concat=False` and `heads=1` (or `hidden_channels // heads` per head) on the single conv is the clean solution.

**Plan 04-03** creates a new `ReverseBackbone` class. When `separate_weights=True` it owns its own `GraphBackbone`; when `False` it holds a reference to the forward backbone passed in at construction. The `project_embeddings` flag adds a `Linear(2 * final_dim, hidden_channels)` layer. The class is standalone — it does not interact with `HashiGraphModel` yet (Phase 5 handles wiring).

**Primary recommendation:** Both new classes live in new files under `src2/hashi_puzzle_solver/models/` to avoid any write conflict with existing files and to match the Phase 4 constraint of touching different files per plan.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | project dependency | Module base, `nn.Module`, `nn.Linear`, `nn.LayerNorm` | All existing model code |
| `torch_geometric.nn.TransformerConv` | project dependency | Attention-based message passing | Already used in `GraphBackbone` |
| `torch.nn.ModuleList` | stdlib | Parameter registration when multiple layers needed | Pattern in `GraphBackbone` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.nn.functional` | stdlib | `relu`, `dropout` | Inside `forward()` for activation |
| `torch.nn.LayerNorm` | stdlib | Post-conv normalization | Used in `GraphBackbone` after every conv |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Single TransformerConv shared across K steps | K separate TransformerConvs (ModuleList) | Separate weights = more parameters, harder to train, and contradicts REAS-01 requirement |
| `concat=False, heads=1` for shared-weight conv | `concat=True, heads=N` then project back | concat=True produces `hidden * heads` dim which breaks residual identity; single head with concat=False gives `hidden_channels` directly |

**Installation:** No new dependencies required.

---

## Architecture Patterns

### Recommended Project Structure

The two new files follow existing naming conventions:

```
src2/hashi_puzzle_solver/models/
├── backbone.py              # EXISTING — GraphBackbone
├── iterative_backbone.py    # NEW (04-02) — IterativeBackbone
├── reverse_backbone.py      # NEW (04-03) — ReverseBackbone
├── core.py                  # EXISTING — HashiGraphModel (unchanged in Phase 4)
├── config.py                # EXISTING (unchanged in Phase 4)
└── ...
```

Test files follow the pattern established by Phase 3:

```
tests/
├── test_config_reasoning.py          # Phase 3 — already exists
├── test_iterative_backbone.py        # NEW (04-02)
└── test_reverse_backbone.py          # NEW (04-03)

src2/hashi_puzzle_solver/trainers/
└── diffusion.py                      # MODIFIED (04-01) — add rev-reason elif
```

### Pattern 1: Trainer Mode Dispatch (04-01)

**What:** The `run_epoch` method dispatches to mode-specific data preparation before the shared forward/loss section.

**Current structure (lines 356-381 of `diffusion.py`):**
```python
scales = None  # initialized by Phase 3 fix

if mode == "diff-cont":
    # ... noise injection
elif mode == "flow-blind":
    # ... flow noise injection
else:
    data = inject_noise(batch, ...)  # diff-discrete and others
```

**After 04-01 change — add before `else`:**
```python
elif mode == "rev-reason":
    data = batch  # no noise injection; edges are clean puzzle state
```

The `num_inference_steps_training` loop below is already generic. For `rev-reason`, `num_inference_steps_training` in the YAML is not set (it defaults to `1`), so the loop runs once per batch. The component flag checks (`reasoning.enabled`, `reverse_gnn.enabled`) are validated as routing correctly — they are config attributes, not trainer state.

### Pattern 2: IterativeBackbone (04-02)

**What:** Single TransformerConv applied K times. Residual add if shapes match (same as `GraphBackbone`).

**Key dimension constraint:** The shared conv layer must have `in_channels == out_channels` so residual addition works every iteration. Use `concat=False` with single head, or `concat=True` with `heads` chosen such that `hidden_channels * heads == node_input_dim` — but the simpler solution is `concat=False, heads=1` producing exactly `hidden_channels` output.

**Constructor signature to match:**
```python
class IterativeBackbone(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        steps: int,
        heads: int = 1,
        dropout: float = 0.25,
        edge_dim: int | None = None,
    ):
        super().__init__()
        self.steps = steps
        self.dropout = dropout
        # concat=False ensures output is hidden_channels regardless of heads
        self.conv = TransformerConv(
            hidden_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=False,
        )
        self.norm = LayerNorm(hidden_channels)
        self.final_dim = hidden_channels

    def forward(self, h, edge_index, edge_attr=None):
        for _ in range(self.steps):
            h_in = h
            h = self.conv(h, edge_index, edge_attr=edge_attr)
            h = self.norm(h)
            h = func.relu(h)
            h = func.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in  # residual always valid: shapes match
        return h
```

**Steps=1 identity test:** With `steps=1`, `IterativeBackbone(h).forward()` produces the same result as calling the single `conv` once — this is trivially true by construction. Test: fix seed, instantiate with `steps=1`, run once; compare to manually calling `conv -> norm -> relu -> dropout -> + h_in`.

### Pattern 3: ReverseBackbone (04-03)

**What:** A wrapper that either owns or references a `GraphBackbone` and applies it with a reversed `edge_index`. Optionally projects concatenated `[forward_h, reverse_h]` to `hidden_channels`.

**Constructor signature:**
```python
class ReverseBackbone(torch.nn.Module):
    def __init__(
        self,
        forward_backbone: GraphBackbone,
        hidden_channels: int,
        separate_weights: bool = True,
        project_embeddings: bool = True,
    ):
        super().__init__()
        self.separate_weights = separate_weights
        self.project_embeddings = project_embeddings

        if separate_weights:
            # Own independent parameters — copy constructor arguments from forward backbone
            self.backbone = GraphBackbone(
                node_input_dim=forward_backbone.convs[0].in_channels,
                hidden_channels=hidden_channels,
                num_layers=len(forward_backbone.convs),
                heads=forward_backbone.convs[0].heads,
                dropout=forward_backbone.dropout,
                edge_dim=forward_backbone.convs[0].edge_dim,
                gnn_type=forward_backbone.gnn_type,
            )
        else:
            # Share weights — hold reference, do NOT register as submodule
            self.backbone = forward_backbone

        self.final_dim = self.backbone.final_dim
        if project_embeddings:
            self.projection = Linear(2 * self.final_dim, hidden_channels)

    def forward(self, h, edge_index, edge_attr=None):
        # Reverse edge direction: flip [2, E] -> swap row 0 and row 1
        rev_edge_index = edge_index.flip(0)
        return self.backbone(h, rev_edge_index, edge_attr=edge_attr)
```

**Note on weight sharing:** When `separate_weights=False`, assigning `self.backbone = forward_backbone` registers it as a submodule (PyTorch `__setattr__` catches `nn.Module`). This means `ReverseBackbone.parameters()` will include the shared backbone parameters, which would double-count them in the optimizer. The correct approach is to store it as a plain attribute, not via `nn.Module` registration:

```python
# Prevents PyTorch from registering it as a submodule:
object.__setattr__(self, "_shared_backbone", forward_backbone)
```

Then in `forward()`: `self._shared_backbone(h, rev_edge_index, ...)`.

Alternatively, accept the double-registration and note it in comments as a known behavior (the optimizer will see duplicates, which is harmless for SGD/Adam since PyTorch `parameters()` deduplicates by identity). Verify the chosen approach's correctness with a test.

### Anti-Patterns to Avoid

- **Adding `IterativeBackbone` to `GraphBackbone` itself:** Phase 4 mandates standalone classes; `HashiGraphModel.forward()` is not touched until Phase 5.
- **`concat=True` with multiple heads in `IterativeBackbone`:** Produces `hidden * heads` output dimension, breaking the residual identity and changing `final_dim` unexpectedly.
- **Modifying `ModelFactory.create_model`:** Phase 4 components are standalone; factory integration is Phase 5.
- **Noise injection in `rev-reason` path:** Explicitly out of scope (REQUIREMENTS.md Out of Scope table). The `else` branch currently handles this; adding a `rev-reason elif` before `else` prevents fallthrough.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Attention message passing | Custom attention | `TransformerConv` from torch_geometric | Already in project, numerically tested, handles multi-head edge attributes |
| MLP layers | Manual `for` loop | `build_mlp()` from `models/common.py` | Already exists in codebase |
| LayerNorm after conv | Custom norm | `torch.nn.LayerNorm` | Established pattern in `GraphBackbone` |

---

## Common Pitfalls

### Pitfall 1: `concat=True` Breaks Residual Identity in IterativeBackbone

**What goes wrong:** If `TransformerConv` is constructed with `concat=True` and `heads=N`, output dimension is `hidden_channels * N`, which does not match `h_in.shape`. The `h = h + h_in` line silently fails (shape error at runtime) or gets skipped.

**Why it happens:** `GraphBackbone` uses `concat=True` for intermediate layers because each layer can change dimension. `IterativeBackbone` must maintain constant dimension.

**How to avoid:** Use `concat=False` unconditionally. Output dimension is then `hidden_channels` regardless of `heads`. Test: `assert iterative_backbone.final_dim == hidden_channels`.

**Warning signs:** `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)` during the residual add.

### Pitfall 2: Double Parameter Registration When Sharing Weights

**What goes wrong:** `self.backbone = forward_backbone` inside `ReverseBackbone.__init__` when `separate_weights=False` registers the shared backbone as a submodule, so `ReverseBackbone.parameters()` includes those weights. If both `HashiGraphModel` and `ReverseBackbone` are registered in the parent model, optimizer sees duplicate parameter references.

**Why it happens:** `nn.Module.__setattr__` intercepts assignment of `nn.Module` instances and adds them to `_modules`.

**How to avoid:** Use `object.__setattr__(self, "_shared_backbone", forward_backbone)` to bypass the `nn.Module` registration. Or store in a plain list: `self._shared_backbone_list = [forward_backbone]` (lists are not scanned for submodules). Test: `assert len(list(reverse_backbone.parameters())) == 0` when `separate_weights=False`.

### Pitfall 3: `rev-reason` Fallthrough to `inject_noise`

**What goes wrong:** If `elif mode == "rev-reason"` is not inserted before the final `else`, the `else` branch calls `inject_noise(batch, ...)` which expects noise-relevant edge feature indices set by the diffusion setup. This could silently corrupt edge features with random noise.

**Why it happens:** The `else` is a catch-all for any unrecognized mode string.

**How to avoid:** Place `elif mode == "rev-reason": data = batch` before the `else`. Test: run `run_epoch` with `mode="rev-reason"` and verify `data.edge_attr` is identical to `batch.edge_attr` (no mutation).

### Pitfall 4: `separate_weights=False` ReverseBackbone Cannot Be Tested in Isolation

**What goes wrong:** Testing `ReverseBackbone(forward_backbone, ..., separate_weights=False)` requires an existing `GraphBackbone` instance, making the test setup more complex. If the forward backbone is in `eval()` mode, the reverse backbone also runs in eval — this is actually correct behavior since they share weights.

**How to avoid:** In tests, always construct a minimal `GraphBackbone` fixture and pass it in. Verify that weight sharing means `id(reverse.backbone)` equals `id(forward)` (or equivalent check using parameter addresses).

---

## Code Examples

Verified patterns from existing codebase:

### TransformerConv constructor (from `backbone.py`)
```python
# Source: src2/hashi_puzzle_solver/models/backbone.py lines 38-46
conv = TransformerConv(
    curr_dim,           # in_channels
    hidden_channels,    # out_channels per head
    heads=out_heads,
    dropout=dropout,
    edge_dim=edge_dim,
    concat=concat,      # concat=False -> output = hidden_channels
)
```

### Residual pattern (from `backbone.py`)
```python
# Source: src2/hashi_puzzle_solver/models/backbone.py lines 83-98
for conv, norm in zip(self.convs, self.norms, strict=True):
    h_in = h
    h = conv(h, edge_index, edge_attr=edge_attr)
    h = norm(h)
    h = func.relu(h)
    h = func.dropout(h, p=self.dropout, training=self.training)
    if h_in.shape == h.shape:
        h = h + h_in
```

### Trainer mode dispatch pattern (from `diffusion.py`)
```python
# Source: src2/hashi_puzzle_solver/trainers/diffusion.py lines 354-381
scales = None

if mode == "diff-cont":
    ...
elif mode == "flow-blind":
    ...
else:
    data = inject_noise(batch, noise_rate, ...)
```

### Config access pattern for nested config (from `diffusion.py`)
```python
# Source: trainers/diffusion.py lines 341-348
training_cfg = self.config["training"]
_bptt = training_cfg.get("bptt", {})
bptt_enabled = (_bptt.get("enabled", False) if isinstance(_bptt, dict) else _bptt.enabled)
mode = training_cfg.get("mode", "diff-discrete").lower()
```

### Accessing model config reasoning/reverse_gnn flags
```python
# config.py: config.model.reasoning.enabled, config.model.reverse_gnn.enabled
# In trainer context where config is dict:
_model_cfg = self.config["model"]
reasoning_enabled = _model_cfg.get("reasoning", {}).get("enabled", False)
# OR if using the typed config object:
self.model_config.model.reasoning.enabled
```

---

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|------------------|-------|
| Noise injection for all modes | Mode-specific dispatch (`diff-cont`, `flow-blind`, `else`) | `rev-reason` becomes the fourth branch |
| Single forward backbone | Optional reverse backbone (separate or shared weights) | Phase 5 integrates; Phase 4 creates the class |

---

## Open Questions

1. **`IterativeBackbone` `heads` parameter**
   - What we know: `GraphBackbone` uses `heads=8` by default; `concat=False` with `heads=1` is simplest for shared-weight iterative conv
   - What's unclear: Whether using `heads > 1` with `concat=False` in `IterativeBackbone` is desirable (it averages multi-head outputs, not necessarily worse)
   - Recommendation: Default `heads=1` for maximum simplicity; caller can override. Document that `concat=False` is fixed.

2. **`ReverseBackbone` `edge_attr` handling when reversing edges**
   - What we know: `edge_attr` is per-edge (shape `[E, D]`). Reversing `edge_index` flips direction but `edge_attr` stays in original edge order.
   - What's unclear: Whether edge attributes (e.g., distance, direction flags) are directional and should also be flipped/permuted.
   - Recommendation: For Phase 4 standalone class, pass `edge_attr` as-is (same attribute tensor). Phase 5 integration can address directionality if needed. Document in class docstring.

3. **Where in `diffusion.py` to check `reasoning.enabled` / `reverse_gnn.enabled` (04-01)**
   - What we know: Phase 4 goal is that the dispatch path exists and component flags route independently with no error; the actual component call is Phase 5
   - What's unclear: 04-01 may just need to establish `data = batch` and leave the forward call unchanged (model will ignore components until Phase 5 wires them in)
   - Recommendation: 04-01 adds the `elif` branch and a clear `# TODO(phase-5): wire IterativeBackbone and ReverseBackbone` comment. No additional flag-checking logic needed in the trainer for Phase 4.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` with `testpaths = ["tests"]`, `pythonpath = ["src2"]` |
| Quick run command | `pytest tests/test_iterative_backbone.py tests/test_reverse_backbone.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

Note: `tests_src2/` is a second test tree. Per `pyproject.toml`, the default `testpaths = ["tests"]`. Run both trees with `pytest tests/ tests_src2/ -x -q`.

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODE-01 | `rev-reason` sets `data = batch` with no noise mutation | unit | `pytest tests/test_diffusion_rev_reason.py::test_rev_reason_no_noise_injection -x` | Wave 0 |
| MODE-02 | Both component flags independently set without error in rev-reason path | unit | `pytest tests/test_diffusion_rev_reason.py::test_rev_reason_component_flags_independent -x` | Wave 0 |
| REAS-01 | `IterativeBackbone` applies single conv K times with residual | unit | `pytest tests/test_iterative_backbone.py::test_iterative_backbone_applies_k_times -x` | Wave 0 |
| REAS-02 | Output shape matches input; steps controls iteration count | unit | `pytest tests/test_iterative_backbone.py::test_iterative_backbone_steps_parameter -x` | Wave 0 |
| REAS-01+02 | `steps=1` output identical to single non-iterative forward | unit | `pytest tests/test_iterative_backbone.py::test_steps_one_matches_single_pass -x` | Wave 0 |
| REVG-01 | `ReverseBackbone.forward()` returns embeddings of same shape as input `h` | unit | `pytest tests/test_reverse_backbone.py::test_reverse_backbone_output_shape -x` | Wave 0 |
| REVG-02 | `separate_weights=True` gives independent params; `=False` shares | unit | `pytest tests/test_reverse_backbone.py::test_separate_weights_independence -x` | Wave 0 |
| REVG-03 | `project_embeddings=True` adds linear projection to `hidden_channels` | unit | `pytest tests/test_reverse_backbone.py::test_project_embeddings_output_dim -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_iterative_backbone.py tests/test_reverse_backbone.py tests/test_diffusion_rev_reason.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_iterative_backbone.py` — covers REAS-01, REAS-02
- [ ] `tests/test_reverse_backbone.py` — covers REVG-01, REVG-02, REVG-03
- [ ] `tests/test_diffusion_rev_reason.py` — covers MODE-01, MODE-02

*(Existing test infrastructure in `tests/` and `tests_src2/` covers all prior phases — no framework install needed.)*

---

## Sources

### Primary (HIGH confidence)
- Direct codebase read — `src2/hashi_puzzle_solver/models/backbone.py` — TransformerConv constructor, residual pattern, `final_dim` property
- Direct codebase read — `src2/hashi_puzzle_solver/trainers/diffusion.py` — mode dispatch structure, `scales = None` placement, forward loop
- Direct codebase read — `src2/hashi_puzzle_solver/models/config.py` — `ReasoningConfig`, `ReverseGnnConfig`, `ModelConfig` fields (Phase 3 complete)
- Direct codebase read — `src2/hashi_puzzle_solver/models/core.py` — `HashiGraphModel.forward()` signature, backbone call site
- Direct codebase read — `src2/hashi_puzzle_solver/models/factory.py` — how backbone is constructed and `final_dim` flows to heads
- Direct codebase read — `pyproject.toml` — test runner configuration, `testpaths`, `pythonpath`

### Secondary (MEDIUM confidence)
- PyTorch `nn.Module.__setattr__` behavior for submodule registration — standard PyTorch behavior, well-documented, applied to `separate_weights=False` analysis

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use in the project
- Architecture: HIGH — derived from reading existing source files
- Pitfalls: HIGH — derived from code analysis (dimension math, PyTorch module registration semantics)

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (stable codebase; no external dependencies changing)
