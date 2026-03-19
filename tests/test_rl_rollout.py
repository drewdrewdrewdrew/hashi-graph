"""Tests for batched RL rollout (Chunk 3)."""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch_geometric.data import Data

from hashi_puzzle_solver.models.transformer import TransformerEdgeClassifier
from hashi_puzzle_solver.rl.config import RLConfig
from hashi_puzzle_solver.rl.env import HashiEnv
from hashi_puzzle_solver.rl.rollout import collect_rollout, random_policy_rollout

# ── helpers ───────────────────────────────────────────────────────────────────

_TERMINAL_REASONS = frozenset({
    "solved",
    "oracle_failure",
    "capacity_failure",
    "crossing_failure",
    "max_steps",
    "no_legal_actions",
})


def _make_puzzle(
    node_capacities: list[int],
    fwd_edges: list[tuple[int, int]],
    targets: list[int],
) -> Data:
    """Construct a minimal bidirectional Hashi puzzle Data object.

    Node features: 3 columns — capacity, structural_degree, unused_capacity.
    Edge features: 4 columns — inv_dx placeholder, inv_dy placeholder,
    is_meta (0), current_bridge_count (0).
    """
    m = len(fwd_edges)
    n = len(node_capacities)

    all_src = [s for s, _ in fwd_edges] + [d for _, d in fwd_edges]
    all_dst = [d for _, d in fwd_edges] + [s for s, _ in fwd_edges]
    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_mask = torch.ones(2 * m, dtype=torch.bool)
    y = torch.tensor(targets + targets, dtype=torch.long)

    deg: Counter[int] = Counter()
    for s, d in fwd_edges:
        deg[s] += 1
        deg[d] += 1

    x_rows = [[float(c), float(deg.get(i, 0)), float(c)] for i, c in enumerate(node_capacities)]
    x = torch.tensor(x_rows, dtype=torch.float)
    node_type = torch.tensor(node_capacities, dtype=torch.long)

    edge_attr = torch.zeros((2 * m, 4), dtype=torch.float)
    edge_attr[:, 0] = 0.5  # inv_dx placeholder
    edge_attr[:, 1] = 0.5  # inv_dy placeholder

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mask=edge_mask,
        y=y,
        node_type=node_type,
        edge_conflict_index=torch.empty((2, 0), dtype=torch.long),
    )


def _make_minimal_model() -> TransformerEdgeClassifier:
    """Return a small randomly-initialised model compatible with test puzzles.

    Compatible with 3-column node features (capacity, degree, unused) and
    4-column edge attributes processed by RLEdgeEncoder.
    """
    torch.manual_seed(0)
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


def _reset_envs(
    puzzles: list[Data],
    config: RLConfig | None = None,
) -> list[HashiEnv]:
    """Create and reset one HashiEnv per puzzle."""
    cfg = config or RLConfig()
    envs = [HashiEnv(cfg) for _ in puzzles]
    for env, puzzle in zip(envs, puzzles):
        env.reset(puzzle)
    return envs


# ── tests ─────────────────────────────────────────────────────────────────────


def test_random_policy_terminates_all_puzzles() -> None:
    """random_policy_rollout terminates all puzzles (solved or failure)."""
    puzzles = [
        _make_puzzle([2, 2], [(0, 1)], [2]),
        _make_puzzle([1, 1, 1, 1], [(0, 1), (2, 3)], [1, 1]),
    ]
    envs = _reset_envs(puzzles)

    results = random_policy_rollout(envs, max_steps=50)

    assert len(results) == len(puzzles)
    for i, (env, r) in enumerate(zip(envs, results)):
        assert env.done, f"env {i} not done after rollout"
        assert r["terminal_reason"] in _TERMINAL_REASONS, (
            f"unexpected terminal_reason={r['terminal_reason']!r} for env {i}"
        )


def test_collect_rollout_random_model_terminates_all() -> None:
    """A random-weights model terminates all envs within max_steps."""
    puzzles = [
        _make_puzzle([2, 2], [(0, 1)], [2]),
        _make_puzzle([1, 1, 1, 1], [(0, 1), (2, 3)], [1, 1]),
        _make_puzzle([2, 1, 2], [(0, 1), (1, 2)], [1, 1]),
    ]
    envs = _reset_envs(puzzles)
    model = _make_minimal_model()

    collect_rollout(envs, model, max_steps=50)

    for i, env in enumerate(envs):
        assert env.done, f"env {i} not done after collect_rollout"


def test_collect_rollout_one_trajectory_per_puzzle() -> None:
    """collect_rollout returns exactly one list per puzzle; lengths ≤ max_steps."""
    puzzles = [
        _make_puzzle([2, 2], [(0, 1)], [2]),
        _make_puzzle([1, 1, 1, 1], [(0, 1), (2, 3)], [1, 1]),
    ]
    envs = _reset_envs(puzzles)
    model = _make_minimal_model()
    max_steps = 20

    trajectories = collect_rollout(envs, model, max_steps=max_steps)

    assert len(trajectories) == len(puzzles)
    for traj in trajectories:
        assert isinstance(traj, list)
        assert len(traj) <= max_steps
        for log_prob, reward in traj:
            assert isinstance(log_prob, torch.Tensor)
            assert log_prob.shape == ()
            assert isinstance(reward, float)


def test_collect_rollout_active_only() -> None:
    """Envs that terminate early stop accumulating trajectory entries."""
    config = RLConfig()
    # env0: single forward edge, target=1 → terminates in exactly 1 rollout step
    # (either solved or oracle_failure — both terminate immediately)
    puzzle1 = _make_puzzle([1, 1], [(0, 1)], [1])
    # env1: two forward edges, target=[1,1] → target_total_bridges=2
    puzzle2 = _make_puzzle([2, 2, 2], [(0, 1), (1, 2)], [1, 1])

    env0 = HashiEnv(config)
    env1 = HashiEnv(config)
    env0.reset(puzzle1)
    env1.reset(puzzle2)

    model = _make_minimal_model()
    trajectories = collect_rollout([env0, env1], model, max_steps=10)

    # env0 must have received exactly 1 trajectory entry (1 forward edge → done in 1 step)
    assert len(trajectories[0]) == 1, (
        f"expected exactly 1 step for env0, got {len(trajectories[0])}"
    )
    assert env0.done

    # env1 must also have terminated (max_steps=10 >> target_total_bridges=2)
    assert env1.done


def test_collect_rollout_log_prob_has_grad() -> None:
    """Each log_prob in a trajectory has a gradient (required for REINFORCE)."""
    puzzles = [
        _make_puzzle([2, 2], [(0, 1)], [2]),
    ]
    envs = _reset_envs(puzzles)
    model = _make_minimal_model()

    trajectories = collect_rollout(envs, model, max_steps=10)

    all_log_probs = [lp for traj in trajectories for lp, _ in traj]
    assert len(all_log_probs) > 0, "no steps were collected — cannot check gradients"
    for lp in all_log_probs:
        assert lp.grad_fn is not None, "log_prob must have a gradient for REINFORCE"
