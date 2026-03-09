---
phase: 05-integration
plan: 01
type: tdd
wave: 1
depends_on: []
files_modified:
  - tests/test_hashi_graph_model_integration.py
  - src2/hashi_puzzle_solver/models/core.py
  - src2/hashi_puzzle_solver/models/factory.py
autonomous: true
requirements: []

must_haves:
  truths:
    - "With all flags disabled, HashiGraphModel.forward() produces byte-for-byte identical output to the pre-phase baseline (backbone call unchanged, no new code paths activated)"
    - "With reasoning.enabled only, IterativeBackbone runs K iterations with residual updates; EdgeHead receives backbone.final_dim"
    - "With reverse_gnn.enabled only, ReverseBackbone output is concatenated with forward embeddings; EdgeHead receives correct dim (hidden_channels if project_embeddings=True, else 2*final_dim)"
    - "With both flags enabled (rev-reasoning), each reasoning iteration interleaves forward conv + reverse pass + projection before the residual update"
    - "EdgeHead receives the correct node_hidden_dim in all four flag combinations without a shape mismatch RuntimeError"
  artifacts:
    - path: "tests/test_hashi_graph_model_integration.py"
      provides: "Integration tests for all 4 success criteria"
      exports:
        - test_flags_disabled_baseline
        - test_reasoning_only
        - test_reverse_only_with_projection
        - test_reverse_only_no_projection
        - test_both_flags_enabled
        - test_rev_reasoning_interleaved
        - test_edge_head_dim_all_combos
    - path: "src2/hashi_puzzle_solver/models/core.py"
      provides: "HashiGraphModel with optional iterative_backbone and reverse_backbone submodules"
      contains: "iterative_backbone, reverse_backbone constructor args; interleaved composition block in forward()"
    - path: "src2/hashi_puzzle_solver/models/factory.py"
      provides: "ModelFactory that builds optional components and computes edge_head_node_dim"
      contains: "edge_head_node_dim computation; IterativeBackbone and ReverseBackbone construction"
  key_links:
    - from: "src2/hashi_puzzle_solver/models/factory.py"
      to: "src2/hashi_puzzle_solver/models/core.py"
      via: "iterative_backbone= and reverse_backbone= constructor kwargs"
      pattern: "HashiGraphModel\\(.*iterative_backbone"
    - from: "src2/hashi_puzzle_solver/models/factory.py"
      to: "src2/hashi_puzzle_solver/models/heads.py"
      via: "edge_head_node_dim computed before EdgeHead construction"
      pattern: "edge_head_node_dim"
    - from: "src2/hashi_puzzle_solver/models/core.py forward()"
      to: "self.iterative_backbone.conv / self.reverse_backbone"
      via: "interleaved loop when both enabled; None-guard when only one or neither"
      pattern: "if self.iterative_backbone is not None"
---

<objective>
Wire IterativeBackbone and ReverseBackbone into HashiGraphModel.forward() via ModelFactory, handle EdgeHead variable input dimensions across all flag combinations, and verify end-to-end composability.

Purpose: Phase 4 delivered standalone components. Phase 5 connects them so the full rev-reason training mode is end-to-end functional and verifiable against all four Phase 5 success criteria.
Output: Updated core.py (optional submodule support), updated factory.py (dim math + conditional construction), new integration test suite that gates the work.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/04-component-implementation/04-component-implementation-01-SUMMARY.md
@.planning/phases/04-component-implementation/04-component-implementation-02-SUMMARY.md
@.planning/phases/04-component-implementation/04-component-implementation-03-SUMMARY.md
</context>

<interfaces>
<!-- Key types and contracts the executor needs. Extracted from codebase. No codebase exploration needed. -->

From src2/hashi_puzzle_solver/models/iterative_backbone.py:
```python
class IterativeBackbone(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        steps: int,
        heads: int = 1,
        dropout: float = 0.25,
        edge_dim: int | None = None,
    ) -> None: ...

    # Public attributes safe to access in the interleaved loop:
    self.steps: int
    self.dropout: float
    self.conv: TransformerConv        # call: self.conv(h, edge_index, edge_attr=...)
    self.norm: LayerNorm              # call: self.norm(h)
    self.final_dim: int               # == hidden_channels (output dim preserved)

    def forward(self, h, edge_index, edge_attr=None) -> torch.Tensor:
        # Applies conv+norm+relu+dropout+residual self.steps times. Output shape == input shape.
```

From src2/hashi_puzzle_solver/models/reverse_backbone.py:
```python
class ReverseBackbone(torch.nn.Module):
    def __init__(
        self,
        forward_backbone: GraphBackbone,
        hidden_channels: int,
        separate_weights: bool = True,
        project_embeddings: bool = True,
    ) -> None: ...

    self.project_embeddings: bool     # True if projection layer exists
    self.final_dim: int               # matches backbone being used
    self.projection: Linear           # Linear(2 * final_dim, hidden_channels) — only if project_embeddings=True

    def forward(self, h, edge_index, edge_attr=None) -> torch.Tensor:
        # Runs backbone on reversed edges (edge_index.flip(0)). Returns [N, final_dim].
        # Concatenation and projection are done by HashiGraphModel.forward(), NOT here.
```

From src2/hashi_puzzle_solver/models/core.py (current state — to be updated):
```python
class HashiGraphModel(torch.nn.Module):
    def __init__(
        self,
        config: HashiModelConfig,
        node_encoder: NodeEncoder,
        edge_encoder: EdgeEncoder,
        backbone: GraphBackbone,
        edge_head: EdgeHead,
        prophet_head: ProphetHead | None = None,
        verify_head: torch.nn.Module | None = None,
        # PHASE 5 adds:
        # iterative_backbone: IterativeBackbone | None = None,
        # reverse_backbone: ReverseBackbone | None = None,
    ): ...

    def forward(self, x, edge_index, edge_attr=None, edge_type=None, batch=None,
                node_type=None, return_verification=False, return_noise=False,
                input_noise=None, time=None, **_kwargs):
        # Current step 6 (to be replaced with composition block):
        h = self.backbone(h, edge_index, edge_attr=h_edge)
        # Step 7 (unchanged):
        edge_logits = self.edge_head(h, edge_index, edge_attr=h_edge, ...)
```

From src2/hashi_puzzle_solver/models/factory.py (current state — to be updated):
```python
class ModelFactory:
    @staticmethod
    def create_model(config: HashiModelConfig, device: torch.device) -> HashiGraphModel:
        # ... builds backbone, then:
        backbone_edge_dim = edge_attr_dim
        if model_config.use_noise_in_message_passing:
            backbone_edge_dim += model_config.noise_embedding_dim
        backbone = GraphBackbone(...)

        # Line 52-56 — TO CHANGE:
        edge_head = EdgeHead(
            model_config,
            node_hidden_dim=backbone.final_dim,   # <-- must become edge_head_node_dim
            edge_attr_dim=backbone_edge_dim
        )
        # prophet_head uses backbone.final_dim — update to edge_head_node_dim too

        model = HashiGraphModel(
            config=config,
            node_encoder=node_encoder,
            edge_encoder=edge_encoder,
            backbone=backbone,
            edge_head=edge_head,
            prophet_head=prophet_head
            # PHASE 5: + iterative_backbone=iterative_bb, reverse_backbone=reverse_bb
        )
```

From src2/hashi_puzzle_solver/models/heads.py (unchanged — interface only):
```python
class EdgeHead(torch.nn.Module):
    def __init__(self, config: ModelConfig, node_hidden_dim: int, edge_attr_dim: int):
        input_dim = 2 * node_hidden_dim  # base: src + dst node embeddings
        # + optional: global_meta, component_meta, edge_features, noise_emb

class ProphetHead(torch.nn.Module):
    def __init__(self, config: ModelConfig, node_hidden_dim: int): ...
```

From src2/hashi_puzzle_solver/models/config.py (Phase 3 deliverable):
```python
# config.model.reasoning.enabled: bool  (default False)
# config.model.reasoning.steps: int     (default 5)
# config.model.reverse_gnn.enabled: bool           (default False)
# config.model.reverse_gnn.separate_weights: bool  (default True)
# config.model.reverse_gnn.project_embeddings: bool (default True)
```
</interfaces>

<tasks>

<task type="auto" tdd="true">
  <name>Task 0: Write failing integration test scaffold (Wave 0 — RED)</name>
  <files>tests/test_hashi_graph_model_integration.py</files>
  <behavior>
    - test_flags_disabled_baseline: Build model with both flags disabled; capture baseline output; rebuild model and run again; assert torch.allclose(out1, out2, atol=0) (deterministic eval). Verifies SC-2.
    - test_reasoning_only: Build model with reasoning.enabled=True, reverse_gnn.enabled=False, steps=2; run forward; assert output.shape == (n_nodes,) or edge logits shape matches edge count. Verifies SC-1 partial.
    - test_reverse_only_with_projection: Build model with reasoning.enabled=False, reverse_gnn.enabled=True, project_embeddings=True; run forward; no RuntimeError. Verifies SC-4 partial.
    - test_reverse_only_no_projection: Build model with reasoning.enabled=False, reverse_gnn.enabled=True, project_embeddings=False; run forward; no RuntimeError. Verifies SC-4 partial (2*final_dim path).
    - test_both_flags_enabled: Build model with both enabled, project_embeddings=True; run forward; no RuntimeError; output shape correct. Verifies SC-1.
    - test_rev_reasoning_interleaved: Build model with both enabled, steps=3; spy on reverse_backbone.forward to count calls; assert it was called exactly steps=3 times (once per reasoning iteration). Verifies SC-3.
    - test_edge_head_dim_all_combos: Parametrize over (reasoning=F,reverse=F), (reasoning=T,reverse=F), (reasoning=F,reverse=T,proj=T), (reasoning=F,reverse=T,proj=F), (reasoning=T,reverse=T,proj=T); assert each runs without shape error. Verifies SC-4.
  </behavior>
  <action>
    Create tests/test_hashi_graph_model_integration.py with all 7 test functions listed above. Tests must FAIL at this stage (HashiGraphModel does not yet accept iterative_backbone/reverse_backbone args, and the interleaved logic does not exist).

    Test fixture helper _make_model(reasoning_enabled, reverse_gnn_enabled, steps=2, project_embeddings=True, separate_weights=True):
    - Build a minimal HashiModelConfig. Reuse the config pattern from tests/test_diffusion_rev_reason.py for the outer config shape. For ModelConfig, set hidden_channels=16, num_layers=2, heads=1, dropout=0.0, type="transformer", and enable only what the test needs.
    - Call ModelFactory.create_model(config, device=torch.device("cpu")).
    - Return the model.

    Test fixture helper _make_batch(n_nodes=6, n_edges=8, hidden_channels=16):
    - Return a dict with x (node features), edge_index, edge_attr, batch tensors suitable for HashiGraphModel.forward().

    Use torch.manual_seed(0) in each test for reproducibility.

    For test_rev_reasoning_interleaved: after building model with both flags enabled and steps=3, wrap model.reverse_backbone.forward with a call-counting spy (same pattern as test_iterative_backbone.py — reassign the method, do NOT use patch.object). Call model.forward(). Assert spy_call_count == 3.

    For test_flags_disabled_baseline: build model with both flags disabled; put in eval(); run forward twice with the same seed; assert torch.equal(out1, out2). This will be green immediately. The real backward-compat test (vs. pre-phase baseline) is enforced by inspecting that the composition block is fully gated behind None checks.

    Commit message: "test(05-01): add failing integration tests for HashiGraphModel composition (Wave 0 RED)"
  </action>
  <verify>
    <automated>cd /mnt/data/user_profiles/andrew/hashi-graph && pytest tests/test_hashi_graph_model_integration.py -x -q 2>&1 | tail -20</automated>
  </verify>
  <done>
    File tests/test_hashi_graph_model_integration.py exists with all 7 test functions. Most tests FAIL (TypeError or AssertionError) because HashiGraphModel does not yet accept optional backbone args. test_flags_disabled_baseline may pass (baseline model is unchanged). Failure is the expected RED state.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 1: Wire optional backbones into HashiGraphModel and ModelFactory (GREEN)</name>
  <files>
    src2/hashi_puzzle_solver/models/core.py
    src2/hashi_puzzle_solver/models/factory.py
  </files>
  <behavior>
    - HashiGraphModel.__init__ accepts two new optional args; stores them as normal nn.Module attributes (NOT object.__setattr__)
    - HashiGraphModel.forward() step 6 becomes a composition block gated entirely by None checks; when both are None the output is byte-for-byte identical to the pre-phase baseline
    - When both iterative_backbone and reverse_backbone are not None, the interleaved loop runs: for each step, call iterative_backbone.conv + norm + relu + dropout, call reverse_backbone, cat + project, residual add
    - ModelFactory builds IterativeBackbone and/or ReverseBackbone when their config flags are enabled
    - ModelFactory computes edge_head_node_dim before constructing EdgeHead; passes it instead of backbone.final_dim
    - If reasoning.enabled and reverse_gnn.enabled and not reverse_gnn.project_embeddings: raise ValueError at HashiGraphModel.__init__ time (residual would be impossible)
  </behavior>
  <action>
    **core.py changes:**

    1. Add imports at top: `from .iterative_backbone import IterativeBackbone` and `from .reverse_backbone import ReverseBackbone`. Use TYPE_CHECKING guard if needed to avoid circular import risk, but direct imports should be fine (no cycles in this module graph).

    2. Add two optional constructor parameters to HashiGraphModel.__init__:
       ```python
       iterative_backbone: IterativeBackbone | None = None,
       reverse_backbone: ReverseBackbone | None = None,
       ```

    3. In __init__ body, after all existing assignments:
       ```python
       self.iterative_backbone = iterative_backbone   # registered as submodule (nn.Module or None)
       self.reverse_backbone = reverse_backbone

       # Validate interleaved constraint at construction time
       if iterative_backbone is not None and reverse_backbone is not None:
           if not reverse_backbone.project_embeddings:
               raise ValueError(
                   "When both reasoning and reverse_gnn are enabled, "
                   "project_embeddings must be True (required for residual dimension match). "
                   "Set model.reverse_gnn.project_embeddings: true in config."
               )
       ```

    4. Replace step 6 in forward() with the composition block:

       ```python
       # 6. Message Passing (Backbone) + optional composition
       h = self.backbone(h, edge_index, edge_attr=h_edge)

       if self.iterative_backbone is not None and self.reverse_backbone is not None:
           # Interleaved rev-reasoning: forward conv + reverse pass + project before each residual
           import torch.nn.functional as F  # already available via func alias — use existing import
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

       IMPORTANT: Check existing imports in core.py. It imports torch but not torch.nn.functional. Add `import torch.nn.functional as F` at the top if not present (check first — do not duplicate).

    **factory.py changes:**

    5. Add imports: `from .iterative_backbone import IterativeBackbone` and `from .reverse_backbone import ReverseBackbone`.

    6. After the backbone is built (after the existing `backbone = GraphBackbone(...)` block), insert optional component construction:

       ```python
       # Optional components (Phase 5)
       iterative_bb: IterativeBackbone | None = None
       if model_config.reasoning.enabled:
           iterative_bb = IterativeBackbone(
               hidden_channels=model_config.hidden_channels,
               steps=model_config.reasoning.steps,
               heads=model_config.heads,
               dropout=model_config.dropout,
               edge_dim=backbone_edge_dim,
           )

       reverse_bb: ReverseBackbone | None = None
       if model_config.reverse_gnn.enabled:
           reverse_bb = ReverseBackbone(
               forward_backbone=backbone,
               hidden_channels=model_config.hidden_channels,
               separate_weights=model_config.reverse_gnn.separate_weights,
               project_embeddings=model_config.reverse_gnn.project_embeddings,
           )
       ```

    7. Compute edge_head_node_dim before the EdgeHead constructor call:

       ```python
       # Compute node_hidden_dim for EdgeHead based on active flag combination
       edge_head_node_dim = backbone.final_dim
       if model_config.reverse_gnn.enabled:
           if model_config.reverse_gnn.project_embeddings:
               edge_head_node_dim = model_config.hidden_channels
           else:
               edge_head_node_dim = 2 * backbone.final_dim
       # reasoning.enabled alone does not change node embedding dim
       ```

    8. Update EdgeHead constructor call: replace `node_hidden_dim=backbone.final_dim` with `node_hidden_dim=edge_head_node_dim`.

    9. Update ProphetHead constructor call (if present): replace `node_hidden_dim=backbone.final_dim` with `node_hidden_dim=edge_head_node_dim` for consistency.

    10. Add `iterative_backbone=iterative_bb, reverse_backbone=reverse_bb` to the `HashiGraphModel(...)` constructor call.

    After changes, run the integration tests and fix any issues before committing. Tests MUST be green.

    Commit message: "feat(05-01): wire IterativeBackbone and ReverseBackbone into HashiGraphModel and ModelFactory"
  </action>
  <verify>
    <automated>cd /mnt/data/user_profiles/andrew/hashi-graph && pytest tests/test_hashi_graph_model_integration.py -x -q</automated>
  </verify>
  <done>
    All 7 tests in test_hashi_graph_model_integration.py pass. Full suite `pytest tests/ -x -q` also passes (or pre-existing failures only — no new failures introduced by this plan).
  </done>
</task>

</tasks>

<verification>
After both tasks complete:

1. `pytest tests/test_hashi_graph_model_integration.py -x -q` — all 7 tests green
2. `pytest tests/test_iterative_backbone.py tests/test_reverse_backbone.py -x -q` — Phase 4 tests still green (no regressions)
3. `pytest tests/ -x -q` — no new failures vs. pre-phase baseline (pre-existing failures in legacy tests are acceptable)
4. Manual inspection: `grep -n "if self.iterative_backbone is not None" src2/hashi_puzzle_solver/models/core.py` returns at least one hit (guards in place)
5. Manual inspection: `grep -n "edge_head_node_dim" src2/hashi_puzzle_solver/models/factory.py` returns at least two hits (computed and used)
</verification>

<success_criteria>
Phase 5 complete when all four roadmap success criteria are verified by automated tests:

SC-1: HashiGraphModel.forward() composes IterativeBackbone and ReverseBackbone; reasoning runs K iterations; reverse output concatenated; projection applied if enabled — test_both_flags_enabled PASSES

SC-2: With all flags disabled, output is byte-for-byte identical to pre-phase baseline — test_flags_disabled_baseline PASSES

SC-3: With both enabled, each reasoning iteration uses forward conv + reverse pass + combined projection before residual update — test_rev_reasoning_interleaved PASSES (spy confirms reverse_backbone.forward called K times, not once)

SC-4: EdgeHead receives correct input dimension in all flag combinations — test_edge_head_dim_all_combos PASSES (no shape mismatch RuntimeError in any combo)
</success_criteria>

<output>
After completion, create `.planning/phases/05-integration/05-integration-01-SUMMARY.md` using the summary template at `.claude/get-shit-done/templates/summary.md`.
</output>
