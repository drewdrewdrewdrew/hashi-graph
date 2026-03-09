# Testing Patterns

**Analysis Date:** 2026-03-06

## Test Framework

**Runner:**
- pytest `>=9.0.2`
- Config in `pyproject.toml` under `[tool.pytest.ini_options]`

**Assertion Library:**
- pytest built-in `assert` (no unittest-style `assertEqual`)
- `pytest.approx` for floating-point comparisons: `assert value == pytest.approx(0.5, abs=1e-5)`
- Direct tensor assertions: `assert tensor.shape == (num_edges, 3)`
- `torch.allclose` for tensor value equality: `assert torch.allclose(collected, global_tensor)`

**Run Commands:**
```bash
pytest tests_src2/          # Run active test suite (src2)
pytest tests/               # Run legacy test suite (src)
pytest tests_src2/ -v       # Verbose output
pytest tests_src2/ -k "test_name"  # Run specific test by name
```

## Test File Organization

**Active tests location:** `tests_src2/` — mirrors the `src2/` package structure

**Legacy tests location:** `tests/` — targets `src/` package, kept for reference

**Naming:**
- Test files: `test_<module_or_feature>.py` — e.g., `test_model_shell.py`, `test_losses.py`, `test_diffusion.py`
- Test functions: `test_<behavior_description>()` in snake_case

**conftest.py:**
- `tests_src2/conftest.py` — inserts `src2/` at `sys.path[0]` to shadow the installed `src/` package
- Contains shared `@pytest.fixture` definitions (e.g., `sample_puzzle_data` in `test_diffusion.py`)

**Structure overview:**
```
tests_src2/
├── conftest.py                      # sys.path fix + shared fixtures
├── test_ar_logic.py                 # AR component detection and rewiring
├── test_components.py               # Backbone and heads
├── test_config.py                   # Config loading and roundtrip
├── test_config_schema.py            # Config field validation + parametrize over YAML files
├── test_data.py                     # Data transforms (MakeBidirectional, GridStretch)
├── test_dataloader_sampling.py      # Val sampler reproducibility
├── test_diffusion.py                # Diffusion trainer, noise injection, buffer logic
├── test_encoders.py                 # NodeEncoder, EdgeEncoder
├── test_features.py                 # NodeFeatureManager, EdgeFeatureManager
├── test_losses.py                   # DegreeLoss, CrossingLoss, HashiLossCalculator
├── test_model_shell.py              # HashiGraphModel integration, noise head flags
├── test_noise_projection.py         # Noise embedding and injection paths
├── test_parity.py                   # End-to-end one training step (requires dataset)
└── test_training_start.py           # Smoke test: trainer init + _setup
```

## Test Structure

**Standard function-based tests (dominant pattern):**
```python
def test_degree_loss():
    """Test that DegreeLoss correctly calculates violations."""
    loss_module = DegreeLoss()

    edge_index = torch.tensor([[0, 1], [1, 0]])
    node_capacities = torch.tensor([1, 1])
    edge_mask = torch.tensor([True, True])
    logits = torch.tensor([[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]])

    loss = loss_module(logits, edge_index, node_capacities, edge_mask)
    assert loss.item() < 1e-3
```

**Class-based tests (used in test_data.py only):**
```python
class TestData(unittest.TestCase):
    def test_make_bidirectional(self) -> None:
        """Test bidirectional edge creation transform."""
        ...

if __name__ == "__main__":
    unittest.main()
```
Note: This is the only file using `unittest.TestCase`; function-based pytest style is preferred.

**Parametrized tests** (used in `test_config_schema.py`):
```python
@pytest.mark.parametrize("config_path", get_yaml_configs(), ids=lambda p: p.name)
def test_all_configs_loadable(config_path: pathlib.Path) -> None:
    """Ensure all YAML configs loadable into HashiModelConfig."""
    config = HashiModelConfig.from_yaml(config_path)
    assert config.data is not None
    assert config.model is not None
    assert config.training is not None
```

**Private helpers for test setup:**
```python
def _make_minimal_model_config(use_noise_head: bool = True) -> HashiModelConfig:
    """Return a minimal HashiModelConfig suitable for unit tests."""
    config = HashiModelConfig()
    config.model.node_embedding_dim = 16
    config.model.hidden_channels = 32
    # ... disable all optional features
    return config
```

**Section dividers** used to group related tests within a file:
```python
# ---------------------------------------------------------------------------
# Step-2 invariant: fresh_alphas must always be exactly zero in _prepare_mixed_batch
# ---------------------------------------------------------------------------
```

## Fixtures

**Inline fixtures in test files** (not in conftest.py) for module-specific data:
```python
@pytest.fixture
def sample_puzzle_data():
    """Simple 2-node puzzle fixture."""
    x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, 5), dtype=torch.float)
    y = torch.tensor([1, 1], dtype=torch.long)
    edge_mask = torch.tensor([True, True], dtype=torch.bool)
    node_type = torch.tensor([1, 1], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=y, edge_mask=edge_mask, node_type=node_type)
```

**Shared builder functions** (not fixtures) for complex data setup:
```python
def _make_cont_batch(n: int = 4) -> Batch:
    """Build a minimal PyG Batch of n puzzles with continuous logit edge attrs."""
    ...
    return Batch.from_data_list(puzzles)

def _make_minimal_trainer() -> DiffusionTrainer:
    """Return a DiffusionTrainer with a mock model, no real data needed."""
    ...
    return trainer
```

**conftest.py approach:** Minimal — only the `sys.path` fix is in `conftest.py`. Fixtures are defined in the test files that need them.

## Mocking

**`monkeypatch` (pytest built-in):** Used to swap out real implementations with dummies:
```python
def test_val_sampler_uses_seeded_randperm(monkeypatch):
    monkeypatch.setattr("hashi_puzzle_solver.trainers.base.HashiDataset", _DummyDataset)
    trainer = _DummyTrainer(...)
    loader = trainer.create_dataloader(split="val")
    assert isinstance(loader.sampler, torch.utils.data.SubsetRandomSampler)
```

**`unittest.mock.patch`:** Used for torch random function mocking in transform tests:
```python
with patch("torch.randint") as mock_randint:
    mock_randint.side_effect = [
        torch.tensor([0]),  # axis
        torch.tensor([0]),  # split_idx
        torch.tensor([1]),  # gap_size
    ]
    with patch("torch.rand") as mock_rand:
        mock_rand.return_value = torch.tensor([0.0])
        stretched_data = stretch(data)
```

**Inline mock classes:** Used to stub out heavy dependencies (model, dataset):
```python
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)  # dummy param
        self.use_verification_head = False
        self.use_noise_head = False

    def forward(self, x, edge_index, edge_attr=None, batch=None, node_type=None, **kwargs):
        num_edges = edge_index.size(1)
        logits = torch.randn(num_edges, 3, device=x.device, requires_grad=True)
        ...
        return logits
```

**What gets mocked:**
- `HashiDataset` — swapped with lightweight `_DummyDataset` to avoid filesystem dependency
- Torch random functions (`torch.randint`, `torch.rand`) — for deterministic transform tests
- Model forward pass — inline `MockModel` when testing trainer logic, not model logic
- Trainer internals — monkey-patched attributes (`trainer.model = MockModel()`)

**What is NOT mocked:**
- The actual model components (tested end-to-end in integration tests)
- PyTorch operations (real tensor ops used throughout)
- Configuration loading (real YAML files read in config tests)

## Test Data Strategy

**Minimal synthetic tensors:** Tests create the smallest graph that exercises the behavior:
```python
num_nodes = 10
num_edges = 20
x = torch.zeros((num_nodes, 1))
edge_index = torch.randint(0, num_nodes, (2, num_edges))
edge_attr = torch.randn(num_edges, em.num_edge_feats)
node_type = torch.tensor([9] + [1] * (num_nodes - 1))
```

**Known-value tensors** for loss and correctness tests:
```python
logits = torch.tensor([
    [0.0, 10.0, 0.0],  # predicts 1 bridge with high confidence
    [0.0, 10.0, 0.0],
])
loss = loss_module(logits, edge_index, node_capacities, edge_mask)
assert loss.item() < 1e-3
```

**Real dataset conditional skipping** for integration tests that require the dataset on disk:
```python
raw_dir = Path(config["data"]["root_dir"]) / "raw"
if not raw_dir.exists() or not list(raw_dir.glob("*.json")):
    pytest.skip("Dataset not found, skipping parity test.")
```

## Test Types

**Unit Tests (majority):**
- Test individual classes in isolation: `NodeEncoder`, `EdgeEncoder`, `GraphBackbone`, `DegreeLoss`, `CrossingLoss`, feature managers
- Verify output shapes and known values
- Files: `test_encoders.py`, `test_components.py`, `test_losses.py`, `test_features.py`, `test_noise_projection.py`, `test_config.py`, `test_features.py`

**Integration Tests:**
- Wire multiple components together and test end-to-end forward passes
- `test_model_shell.py` — full `HashiGraphModel` forward pass
- `test_diffusion.py` — `DiffusionTrainer.run_epoch` with mock model
- `test_dataloader_sampling.py` — full dataloader creation with monkeypatched dataset

**Smoke / Setup Tests:**
- Verify initialization completes without error
- `test_training_start.py` — `Trainer._setup()` creates model and optimizer
- `test_config_schema.py` — all YAML configs parse without `TypeError`

**Parity / End-to-End Tests:**
- `test_parity.py` — one real training step requires dataset on disk; skipped when absent
- `test_ar_logic.py` — full batch rewiring with mock puzzle objects

**Invariant Tests:**
- Enforce specific mathematical/algorithmic properties that must hold
- `test_fresh_alphas_are_zero`, `test_fresh_sigmas_equal_sigma_max` in `test_diffusion.py`
- Named with the invariant they guard: docstrings explain the "Step-N invariant"

## Coverage

**Requirements:** Not enforced (no `--cov` or coverage config found)

**View Coverage:**
```bash
pytest tests_src2/ --cov=src2/hashi_puzzle_solver --cov-report=term-missing
```

**Observed gaps:**
- `src2/hashi_puzzle_solver/data.py` (1285 lines) has only partial coverage from `test_data.py` (3 transform tests)
- `src2/hashi_puzzle_solver/trainers/ar.py` — no direct test file in `tests_src2/`
- `src2/hashi_puzzle_solver/tune.py` and `tune_space.py` — no tests in `tests_src2/`
- `src2/hashi_puzzle_solver/callbacks.py` — not directly tested

## Async Testing

Not applicable — no async code in this codebase.

## Common Patterns

**Shape assertion (most common):**
```python
assert logits.shape == (num_edges, 3)
assert aux_logits.shape == (1, 2)  # (sigma, alpha) for one graph
```

**Type assertion for return value variants:**
```python
assert isinstance(result, torch.Tensor), (
    "Expected a plain tensor (no noise tuple) when use_noise_head=False"
)
```

**Dict key presence:**
```python
assert "total" in losses
assert "ce" in losses
assert "degree" in losses
```

**Dataclass field introspection:**
```python
from dataclasses import fields
field_names = {f.name for f in fields(TrainingConfig)}
assert "weight_decay" in field_names
```

**Approximate float equality:**
```python
assert abs(loss.item() - 1.0) < 1e-3
assert stretched_data.edge_attr[0, 0].item() == pytest.approx(0.5, abs=1e-5)
```

---

*Testing analysis: 2026-03-06*
