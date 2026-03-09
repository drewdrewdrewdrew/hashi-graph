# Coding Conventions

**Analysis Date:** 2026-03-06

## Naming Patterns

**Files:**
- `snake_case.py` for all source files: `bridges_gen.py`, `train_utils.py`, `diffusion_utils.py`
- Subpackage grouping by concern: `models/`, `trainers/`, `losses/`, `utils/`
- Legacy files retained in `src/`, active code in `src2/hashi_puzzle_solver/`

**Classes:**
- `PascalCase` throughout: `HashiGraphModel`, `GraphBackbone`, `NodeEncoder`, `EdgeEncoder`, `BaseTrainer`, `DiffusionTrainer`, `HashiModelConfig`
- Abstract base classes suffixed or named clearly: `LossModule` (ABC), `BaseTrainer`
- Data containers use plain names: `EpochMetrics`, `EarlyStopper`

**Functions and methods:**
- `snake_case` throughout: `run_epoch`, `create_dataloader`, `inject_noise`, `check_puzzle_solved`
- Private helpers prefixed with `_`: `_setup`, `_dict_to_metrics`, `_prepare_mixed_batch`, `_refill_buffer`, `_make_minimal_model_config` (in tests)
- Factory methods use `from_*` and `to_*` classmethods: `HashiModelConfig.from_yaml`, `HashiModelConfig.from_dict`, `HashiModelConfig.to_dict`

**Variables:**
- `snake_case` throughout: `edge_index`, `node_type`, `noise_emb`, `hidden_channels`
- Tensor names match their domain: `h` for node hidden state, `h_edge` for edge hidden state, `edge_logits`, `noise_logits`
- Loop indices kept minimal and conventional: `i`, `k`, `v`

**Constants and config fields:**
- All lowercase with underscores in dataclasses: `node_embedding_dim`, `use_noise_head`, `sigma_max`
- Boolean toggles prefixed `use_*`: `use_global_meta_node`, `use_row_col_meta`, `use_noise_in_message_passing`

## Code Style

**Formatter:**
- Ruff format (`ruff>=0.14.10`), configured in `pyproject.toml`
- `quote-style = "double"` — always use double quotes
- `indent-style = "space"` — 4-space indentation (Python default)
- `line-length = 88` (Black-compatible)
- `line-ending = "lf"`
- `skip-magic-trailing-comma = false` — trailing commas respected

**Linter:**
- Ruff lint with an extensive rule set (F, E, W, I, D, UP, ANN, B, C4, PT, RET, SIM, PTH, etc.)
- `preview = true` — enables experimental rules
- `fix = true` — auto-fixes enabled, `unsafe-fixes = false`
- `unfixable = ["ERA001"]` — commented-out code is flagged but never auto-deleted

**Type annotations:**
- Required on all public functions and methods (ANN ruleset enforced)
- Union types use `X | Y` syntax (Python 3.10+ style via pyupgrade)
- Optional parameters typed as `X | None`: `edge_attr: torch.Tensor | None = None`
- Return types always annotated: `-> torch.Tensor`, `-> dict[str, Any]`, `-> bool`
- `from __future__ import annotations` not used; `keep-runtime-typing = false` in pyupgrade
- `typing.Any` used where config dicts have heterogeneous values

## Docstring Style

**Convention:** NumPy docstring style (configured via `[tool.ruff.lint.pydocstyle] convention = "numpy"`)

**Module docstrings:**
- All `src2` modules have a one-line module docstring: `"""Backbone GNN for Hashi Puzzle Solver."""`
- `D100` (missing module docstring) and `D104` (package docstring) are ignored

**Class docstrings:**
- Short summary line on classes: `"""Backbone GNN that performs message passing across the graph."""`
- Multi-line when useful, but no section headers required for simple classes

**Function/method docstrings:**
- One-line summary for simple methods: `"""Encode node features into hidden representations."""`
- NumPy-style `Args:` and `Returns\n-------` sections for public methods with meaningful parameters
- Example from `src2/hashi_puzzle_solver/models/config.py`:
```python
def from_yaml(cls, yaml_path: str | pathlib.Path) -> "HashiModelConfig":
    """
    Load configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns
    -------
        HashiModelConfig: Populated configuration object.
    """
```
- Private/internal helpers may have minimal or no docstrings

## Import Organization

**Order (enforced by isort via Ruff):**
1. Standard library: `pathlib`, `typing`, `dataclasses`, `abc`
2. Third-party: `torch`, `torch_geometric`, `yaml`, `tqdm`, `numpy`
3. First-party (intra-package): relative imports from `.config`, `..utils.common`, `..models.factory`

**Style:**
- `combine-as-imports = true` — `from x import a, b` preferred over multiple lines
- `force-sort-within-sections = true`
- Relative imports used for intra-package references: `from .backbone import GraphBackbone`
- Absolute imports used when crossing package boundary or in scripts

**Example from `src2/hashi_puzzle_solver/trainers/base.py`:**
```python
import torch
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models.config import HashiModelConfig
from ..models.factory import ModelFactory
from ..losses.calculator import HashiLossCalculator
```

## Error Handling

**Strategy:** Raise with clear messages; use f-strings stored in variables (EM ruleset)

**Pattern for invalid inputs:**
```python
msg = f"Unknown GNN type: {gnn_type}"
raise ValueError(msg)
```

**Validation pattern:**
```python
error_msg = "Each edge_conflict must contain exactly two entries."
raise ValueError(error_msg)
```

**No bare `except` clauses observed.** Exceptions caught with specific types where needed (e.g., `except TypeError` in config schema tests).

**Graceful fallbacks preferred over exceptions** for optional config keys:
```python
config["training"].get("adam_epsilon", 1e-8)
```

## Configuration Pattern

**Dataclasses for typed config** (`src2/hashi_puzzle_solver/models/config.py`):
- `@dataclass` with default values for all fields
- Nested composition via `field(default_factory=...)`: `LossWeightsConfig`, `EarlyStoppingConfig` embedded in `TrainingConfig`
- Root config object `HashiModelConfig` has `from_yaml`, `from_dict`, `to_dict` class methods

**Dict-based config also accepted** in trainers (from YAML via `load_config`):
- `BaseTrainer.__init__` accepts `dict[str, Any]` and internally creates `HashiModelConfig.from_dict(config)`
- Both APIs coexist; trainer tests use plain dicts for brevity

## Module Design

**Exports:**
- `__init__.py` files are present but typically sparse; modules import directly from submodules
- No heavy barrel files; consumers import specific symbols: `from hashi_puzzle_solver.models.core import HashiGraphModel`

**Separation of concerns:**
- `models/config.py` — configuration dataclasses only
- `models/features.py` — feature index management (`NodeFeatureManager`, `EdgeFeatureManager`)
- `models/encoders.py` — encoder modules only (depend on feature managers)
- `models/backbone.py` — GNN backbone only
- `models/heads.py` — prediction heads only
- `models/core.py` — assembly shell (`HashiGraphModel`) wires components together
- `models/factory.py` — construction logic (`ModelFactory.create_model`)
- `losses/` — one file per loss type (`degree.py`, `crossing.py`, `verification.py`), unified via `calculator.py`
- `trainers/` — one trainer per training mode (`one_shot.py`, `diffusion.py`, `ar.py`) sharing `base.py`
- `utils/` — domain-specific utilities: `ar_utils.py`, `diffusion_utils.py`, `train_utils.py`, `graph_utils.py`, `common.py`

**`pathlib.Path` enforced** over `os.path` strings (PTH ruleset active)

## Comments

**Inline comments** used for step-by-step annotations in forward passes:
```python
# 1. Encode Nodes
h = self.node_encoder(x)
# 2. Encode Edges
h_edge = self.edge_encoder(edge_attr, edge_type)
```

**Explanatory comments** used for non-obvious decisions and TODOs left in code (ERA001 rule flags but does not auto-delete commented code):
- `# Diffusion specific components`
- `# Skip connection if dimensions match`

**No active TODO/FIXME/HACK/XXX markers found** in `src2/` source files.

---

*Convention analysis: 2026-03-06*
