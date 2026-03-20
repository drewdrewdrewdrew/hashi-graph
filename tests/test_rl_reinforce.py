"""Tests for the REINFORCE training loop (Chunk 4)."""

from __future__ import annotations

from collections import Counter
import math

from hashi_puzzle_solver.rl.config import RLConfig
from hashi_puzzle_solver.rl.reinforce import (
    ReinforceTrainer,
    compute_returns,
    evaluate,
    reinforce_loss,
    train_one_update,
)
import pytest
import torch
from torch_geometric.data import Data

from hashi_puzzle_solver.models.transformer import TransformerEdgeClassifier

# ── helpers ───────────────────────────────────────────────────────────────────

_EXPECTED_METRIC_KEYS = frozenset({
    "perfect_accuracy",
    "edge_acc",
    "avg_episode_length",
    "avg_solve_length",
    "avg_return",
    "oracle_failure_rate",
    "capacity_failure_rate",
    "crossing_failure_rate",
    "dead_end_unsolved_rate",
    "max_steps_rate",
})


def _make_puzzle(
    node_capacities: list[int],
    fwd_edges: list[tuple[int, int]],
    targets: list[int],
) -> Data:
    """Construct a minimal bidirectional Hashi puzzle Data object."""
    m = len(fwd_edges)

    all_src = [s for s, _ in fwd_edges] + [d for _, d in fwd_edges]
    all_dst = [d for _, d in fwd_edges] + [s for s, _ in fwd_edges]
    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_mask = torch.ones(2 * m, dtype=torch.bool)
    y = torch.tensor(targets + targets, dtype=torch.long)

    deg: Counter[int] = Counter()
    for s, d in fwd_edges:
        deg[s] += 1
        deg[d] += 1

    x_rows = [
        [float(c), float(deg.get(i, 0)), float(c)]
        for i, c in enumerate(node_capacities)
    ]
    x = torch.tensor(x_rows, dtype=torch.float)
    node_type = torch.tensor(node_capacities, dtype=torch.long)

    edge_attr = torch.zeros((2 * m, 4), dtype=torch.float)
    edge_attr[:, 0] = 0.5
    edge_attr[:, 1] = 0.5

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mask=edge_mask,
        y=y,
        node_type=node_type,
        edge_conflict_index=torch.empty((2, 0), dtype=torch.long),
    )


def _make_minimal_model(seed: int = 0) -> TransformerEdgeClassifier:
    """Return a small randomly-initialised model compatible with test puzzles."""
    torch.manual_seed(seed)
    return TransformerEdgeClassifier(
        node_embedding_dim=8,
        hidden_channels=16,
        num_layers=1,
        heads=1,
        dropout=0.0,
        edge_dim=8,
        use_capacity=True,
        use_structural_degree=True,
        use_unused_capacity=True,
        use_conflict_status=False,
        use_rl_edge_encoder=True,
        rl_raw_edge_input_dim=4,
    )


# ── compute_returns ────────────────────────────────────────────────────────────


def test_compute_returns_exact_values() -> None:
    """compute_returns produces correct discounted returns for a known sequence."""
    # rewards = [1.0, 1.0, 11.0], gamma = 1.0
    # G_2 = 11.0, G_1 = 1.0 + 11.0 = 12.0, G_0 = 1.0 + 12.0 = 13.0
    returns = compute_returns([1.0, 1.0, 11.0], gamma=1.0)
    assert returns == pytest.approx([13.0, 12.0, 11.0])


def test_compute_returns_discounted() -> None:
    """Discount factor is applied correctly."""
    # rewards = [1.0, 2.0], gamma = 0.9
    # G_1 = 2.0, G_0 = 1.0 + 0.9 * 2.0 = 2.8
    returns = compute_returns([1.0, 2.0], gamma=0.9)
    assert returns == pytest.approx([2.8, 2.0])


def test_compute_returns_single_step() -> None:
    """Single-step trajectory returns the reward itself."""
    returns = compute_returns([5.0], gamma=1.0)
    assert returns == pytest.approx([5.0])


def test_compute_returns_empty() -> None:
    """Empty reward list returns an empty list."""
    assert compute_returns([], gamma=1.0) == []


# ── reinforce_loss ─────────────────────────────────────────────────────────────


def test_reinforce_loss_differentiable() -> None:
    """reinforce_loss.backward() succeeds and sets gradients on log_probs."""
    # Construct a 3-step trajectory with leaf tensors
    lp0 = torch.tensor(-0.7, requires_grad=True)
    lp1 = torch.tensor(-0.4, requires_grad=True)
    lp2 = torch.tensor(-0.9, requires_grad=True)
    traj: list[tuple[torch.Tensor, float]] = [
        (lp0, 1.0),
        (lp1, 1.0),
        (lp2, 11.0),
    ]

    loss = reinforce_loss([traj], gamma=1.0)
    loss.backward()

    for lp in (lp0, lp1, lp2):
        assert lp.grad is not None, "gradient must flow back to log_probs"


def test_reinforce_loss_empty_trajectories() -> None:
    """Empty trajectory list returns a zero loss tensor."""
    loss = reinforce_loss([])
    assert loss.item() == pytest.approx(0.0)
    # Must still be a tensor (not a Python float) so backward() is callable
    assert isinstance(loss, torch.Tensor)


def test_reinforce_loss_multi_trajectory() -> None:
    """Loss is computed over all trajectories jointly."""
    lp0 = torch.tensor(-1.0, requires_grad=True)
    lp1 = torch.tensor(-0.5, requires_grad=True)
    traj_a: list[tuple[torch.Tensor, float]] = [(lp0, 10.0)]
    traj_b: list[tuple[torch.Tensor, float]] = [(lp1, -5.0)]

    loss = reinforce_loss([traj_a, traj_b], gamma=1.0)
    assert math.isfinite(loss.item())
    loss.backward()
    assert lp0.grad is not None
    assert lp1.grad is not None


# ── train_one_update ───────────────────────────────────────────────────────────


def test_train_one_update_returns_loss_dict() -> None:
    """train_one_update returns a dict with a finite float 'loss' key."""
    puzzle = _make_puzzle([1, 2, 1], [(0, 1), (1, 2)], [1, 1])
    model = _make_minimal_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = RLConfig()

    result = train_one_update([puzzle], model, optimizer, config, max_steps=20)

    assert isinstance(result, dict)
    assert "loss" in result
    assert isinstance(result["loss"], float)
    assert math.isfinite(result["loss"])


def test_train_one_update_changes_model_parameters() -> None:
    """Model parameters change after a training update on a multi-action puzzle.

    Uses a batch of puzzles so that cross-puzzle return variance produces a
    non-trivial gradient signal even after baseline subtraction.
    """
    torch.manual_seed(7)
    # 2 forward edges, targets=[1, 1]: each step the model picks from 2+ legal actions
    puzzle = _make_puzzle([1, 2, 1], [(0, 1), (1, 2)], [1, 1])
    puzzles = [puzzle] * 6

    model = _make_minimal_model(seed=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = RLConfig()

    params_before = [p.clone().detach() for p in model.parameters()]

    # Run a handful of updates — at least one should produce a non-zero gradient
    for _ in range(10):
        train_one_update(puzzles, model, optimizer, config, max_steps=20)

    changed = any(
        not torch.allclose(pb, pa)
        for pb, pa in zip(params_before, model.parameters(), strict=True)
    )
    assert changed, "Model parameters must change after REINFORCE updates"


def test_train_one_update_loss_finite_over_many_steps() -> None:
    """Loss stays finite across many training updates (no NaN / Inf blowup)."""
    torch.manual_seed(42)
    puzzle = _make_puzzle([2, 2, 2], [(0, 1), (1, 2)], [1, 1])
    puzzles = [puzzle] * 4

    model = _make_minimal_model(seed=42)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    config = RLConfig()

    for _ in range(20):
        result = train_one_update(puzzles, model, optimizer, config, max_steps=15)
        assert math.isfinite(result["loss"]), (
            f"Loss became non-finite: {result['loss']}"
        )


# ── evaluate ──────────────────────────────────────────────────────────────────


def test_evaluate_returns_all_metric_keys() -> None:
    """Evaluate returns exactly the expected set of metric keys."""
    puzzle = _make_puzzle([1, 2, 1], [(0, 1), (1, 2)], [1, 1])
    model = _make_minimal_model()
    config = RLConfig()

    metrics = evaluate([puzzle], model, config, max_steps=20)

    assert set(metrics.keys()) == _EXPECTED_METRIC_KEYS


def test_evaluate_metric_values_in_range() -> None:
    """Rate/accuracy metrics are in [0, 1]; length/return metrics are non-negative."""
    puzzles = [
        _make_puzzle([2, 2], [(0, 1)], [2]),
        _make_puzzle([1, 2, 1], [(0, 1), (1, 2)], [1, 1]),
        _make_puzzle([2, 1, 2], [(0, 1), (1, 2)], [1, 1]),
    ]
    model = _make_minimal_model()
    config = RLConfig()

    metrics = evaluate(puzzles, model, config, max_steps=20)

    rate_keys = {
        "perfect_accuracy",
        "edge_acc",
        "oracle_failure_rate",
        "capacity_failure_rate",
        "crossing_failure_rate",
        "dead_end_unsolved_rate",
        "max_steps_rate",
    }
    for key in rate_keys:
        assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of [0,1]"

    assert metrics["avg_episode_length"] >= 0.0
    assert metrics["avg_solve_length"] >= 0.0


def test_evaluate_terminal_rates_sum_to_one() -> None:
    """Terminal reason rates plus any unaccounted fraction must sum to ≤ 1."""
    puzzles = [
        _make_puzzle([2, 2], [(0, 1)], [2]),
        _make_puzzle([1, 1], [(0, 1)], [1]),
        _make_puzzle([2, 1, 2], [(0, 1), (1, 2)], [1, 1]),
    ]
    model = _make_minimal_model()
    config = RLConfig()

    metrics = evaluate(puzzles, model, config, max_steps=30)

    total = (
        metrics["perfect_accuracy"]
        + metrics["oracle_failure_rate"]
        + metrics["capacity_failure_rate"]
        + metrics["crossing_failure_rate"]
        + metrics["dead_end_unsolved_rate"]
        + metrics["max_steps_rate"]
    )
    assert total == pytest.approx(1.0, abs=1e-6)


def test_evaluate_empty_puzzle_list() -> None:
    """Evaluate on an empty puzzle list returns zeros without error."""
    model = _make_minimal_model()
    config = RLConfig()

    metrics = evaluate([], model, config)

    for key, val in metrics.items():
        assert val == pytest.approx(0.0), f"{key} should be 0.0 for empty list"


# ── random baseline sanity ────────────────────────────────────────────────────


def test_random_baseline_perfect_accuracy_not_trivially_perfect() -> None:
    """Random-weight model does not achieve 100% perfect_accuracy.

    With randomly initialised weights and a variety of 2- and 3-island
    puzzles, the model's greedy policy will make oracle errors, confirming
    that the metric is sensitive and not trivially 1.0.
    """
    torch.manual_seed(99)
    puzzles = [
        # 3-island chain: two required edges
        _make_puzzle([1, 2, 1], [(0, 1), (1, 2)], [1, 1]),
        _make_puzzle([2, 2, 2], [(0, 1), (1, 2)], [1, 1]),
        _make_puzzle([2, 4, 2], [(0, 1), (1, 2)], [1, 1]),
        _make_puzzle([1, 2, 1], [(0, 1), (1, 2)], [1, 1]),
        _make_puzzle([2, 2, 2], [(0, 1), (1, 2)], [1, 1]),
        _make_puzzle([2, 4, 2], [(0, 1), (1, 2)], [1, 1]),
    ]

    # Use a fresh random model (no training)
    model = _make_minimal_model(seed=99)
    config = RLConfig()

    metrics = evaluate(puzzles, model, config, max_steps=20)

    # With random weights on puzzles that have oracle-failure-prone zero-target
    # edges or ordering-sensitive solutions, perfect accuracy < 1.0 is expected.
    assert metrics["perfect_accuracy"] < 1.0, (
        f"Random model should not achieve 100% accuracy; "
        f"got {metrics['perfect_accuracy']}"
    )


# ── ReinforceTrainer ──────────────────────────────────────────────────────────


def test_reinforce_trainer_interface() -> None:
    """ReinforceTrainer.train() and .evaluate() run without error."""
    torch.manual_seed(3)
    puzzle = _make_puzzle([1, 2, 1], [(0, 1), (1, 2)], [1, 1])
    model = _make_minimal_model(seed=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = RLConfig()

    trainer = ReinforceTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        train_puzzles=[puzzle] * 3,
    )

    history = trainer.train(n_updates=5, max_steps=15)

    assert len(history) == 5
    assert trainer.update_count == 5
    assert len(trainer.loss_history) == 5
    for entry in history:
        assert "loss" in entry
        assert math.isfinite(entry["loss"])

    metrics = trainer.evaluate(max_steps=15)
    assert set(metrics.keys()) == _EXPECTED_METRIC_KEYS
