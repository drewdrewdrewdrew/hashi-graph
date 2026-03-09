"""Tests for the rev-reason mode dispatch in DiffusionTrainer.run_epoch.

Covers MODE-01 (no noise injection in rev-reason path) and MODE-02
(reasoning.enabled and reverse_gnn.enabled flags are independently settable
without any error when mode is rev-reason).

TDD RED: These tests fail before the ``elif mode == "rev-reason"`` branch is
added to trainers/diffusion.py.
"""

from __future__ import annotations

import torch
import pytest
from unittest.mock import patch, MagicMock
from torch_geometric.data import Data, Batch

from hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer


# ---------------------------------------------------------------------------
# Minimal synthetic batch fixture
# ---------------------------------------------------------------------------

def _make_batch(edge_attr_dim: int = 5) -> Batch:
    """Return a single-graph Batch suitable for a rev-reason run_epoch call.

    Node features: [capacity, structural_degree, unused_capacity]
    Edge features: zeros of shape (num_edges, edge_attr_dim)
    """
    torch.manual_seed(0)
    x = torch.tensor([[2.0, 2.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, edge_attr_dim), dtype=torch.float)
    y = torch.tensor([1, 1], dtype=torch.long)
    edge_mask = torch.tensor([True, True], dtype=torch.bool)
    node_type = torch.tensor([1, 1], dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        edge_mask=edge_mask,
        node_type=node_type,
    )
    return Batch.from_data_list([data])


# ---------------------------------------------------------------------------
# Minimal config and trainer factory
# ---------------------------------------------------------------------------

def _make_config(
    mode: str = "rev-reason",
    reasoning_enabled: bool = False,
    reverse_gnn_enabled: bool = False,
) -> dict:
    """Return a minimal dict-based config for DiffusionTrainer."""
    return {
        "data": {},
        "model": {
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_edge_labels_as_features": True,
            "use_verification_head": False,
            "use_noise_head": False,
            "use_component_meta": False,
            "reasoning": {"enabled": reasoning_enabled, "steps": 5},
            "reverse_gnn": {"enabled": reverse_gnn_enabled},
        },
        "training": {
            "mode": mode,
            "learning_rate": 1e-3,
            "loss_weights": {"ce": 1.0},
        },
    }


class MockModel(torch.nn.Module):
    """Minimal model returning zero logits for any edge set."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        self.use_verification_head = False
        self.use_noise_head = False

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return zero logits of shape (num_edges, 3)."""
        num_edges = edge_index.size(1)
        return torch.zeros(num_edges, 3, requires_grad=True)


def _make_trainer(config: dict) -> DiffusionTrainer:
    """Instantiate a DiffusionTrainer with a mock model and optimizer attached."""
    device = torch.device("cpu")
    trainer = DiffusionTrainer(config=config, device=device)
    model = MockModel().to(device)
    trainer.model = model
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return trainer


# ---------------------------------------------------------------------------
# MODE-01: No noise injection in the rev-reason path
# ---------------------------------------------------------------------------

def test_rev_reason_no_noise_injection() -> None:
    """MODE-01: run_epoch with mode=rev-reason must not call any noise injection function.

    Also verifies that data.edge_attr inside the epoch is not mutated —
    the batch is passed directly to the model without noise modification.
    """
    config = _make_config(mode="rev-reason")
    trainer = _make_trainer(config)
    batch = _make_batch()
    original_edge_attr = batch.edge_attr.clone()

    inject_noise_path = "hashi_puzzle_solver.trainers.diffusion.inject_noise"
    inject_cont_path = "hashi_puzzle_solver.trainers.diffusion.inject_continuous_noise"
    inject_flow_path = "hashi_puzzle_solver.trainers.diffusion.inject_flow_noise"

    with (
        patch(inject_noise_path) as mock_noise,
        patch(inject_cont_path) as mock_cont,
        patch(inject_flow_path) as mock_flow,
    ):
        trainer.run_epoch(loader=[batch], epoch=1, total_epochs=1, training=False)

    mock_noise.assert_not_called()
    mock_cont.assert_not_called()
    mock_flow.assert_not_called()

    # Edge attributes on the original batch tensor must be unchanged.
    assert torch.equal(batch.edge_attr, original_edge_attr), (
        "batch.edge_attr was mutated in the rev-reason path"
    )


# ---------------------------------------------------------------------------
# MODE-02: Component flags are independently settable without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("reasoning_enabled", "reverse_gnn_enabled"),
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_rev_reason_component_flags_independent(
    reasoning_enabled: bool, reverse_gnn_enabled: bool
) -> None:
    """MODE-02: All four flag combinations must complete run_epoch without raising.

    The rev-reason elif branch must be reachable regardless of what
    reasoning.enabled and reverse_gnn.enabled are set to.
    """
    config = _make_config(
        mode="rev-reason",
        reasoning_enabled=reasoning_enabled,
        reverse_gnn_enabled=reverse_gnn_enabled,
    )
    trainer = _make_trainer(config)
    batch = _make_batch()

    # Must not raise for any flag combination.
    trainer.run_epoch(loader=[batch], epoch=1, total_epochs=1, training=False)
