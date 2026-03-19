"""Tests for the HashiEnv oracle-aware RL environment (Chunk 1)."""

from collections import Counter

from hashi_puzzle_solver.rl.config import RLConfig
from hashi_puzzle_solver.rl.env import HashiEnv
import pytest
import torch
from torch_geometric.data import Data

# ── test fixture helpers ──────────────────────────────────────────────────────

_DEFAULT_MODEL_CFG = {
    "use_capacity": True,
    "use_structural_degree": True,
    "use_unused_capacity": True,
}


def _make_puzzle(
    node_capacities: list[int],
    fwd_edges: list[tuple[int, int]],
    targets: list[int],
    conflict_pairs: list[tuple[int, int]] | None = None,
) -> Data:
    """Construct a minimal bidirectional Hashi puzzle Data object.

    Parameters
    ----------
    node_capacities : list[int]
        Island capacity (= ``node_type``) for each node, 1-8.
    fwd_edges : list[tuple[int, int]]
        Forward directed edges ``(src, dst)``.  The reverse edges
        ``(dst, src)`` are appended automatically so the graph is
        bidirectional with ``M = len(fwd_edges)``.
    targets : list[int]
        Target bridge count for each forward edge (0, 1, or 2).
    conflict_pairs : list[tuple[int, int]] or None
        Pairs of *forward* edge indices that cross each other.

    Returns
    -------
    Data
        Puzzle graph with ``x`` (3 node features), ``edge_index``,
        ``edge_attr`` (4 columns), ``edge_mask``, ``y``, ``node_type``,
        and ``edge_conflict_index``.
    """
    m = len(fwd_edges)

    # bidirectional edges: forward then reverse
    all_src = [s for s, _ in fwd_edges] + [d for _, d in fwd_edges]
    all_dst = [d for _, d in fwd_edges] + [s for s, _ in fwd_edges]
    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)

    # all edges are puzzle edges
    edge_mask = torch.ones(2 * m, dtype=torch.bool)

    # targets for both directions (same value)
    y = torch.tensor(targets + targets, dtype=torch.long)

    # structural degree per node
    deg: Counter[int] = Counter()
    for s, d in fwd_edges:
        deg[s] += 1
        deg[d] += 1

    x_rows = []
    for i, cap in enumerate(node_capacities):
        x_rows.append([float(cap), float(deg.get(i, 0)), float(cap)])
    x = torch.tensor(x_rows, dtype=torch.float)

    node_type = torch.tensor(node_capacities, dtype=torch.long)

    edge_attr = torch.zeros((2 * m, 4), dtype=torch.float)
    edge_attr[:, 0] = 0.5
    edge_attr[:, 1] = 0.5

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mask=edge_mask,
        y=y,
        node_type=node_type,
    )

    if conflict_pairs:
        data.edge_conflict_index = torch.tensor(
            [[a for a, _ in conflict_pairs], [b for _, b in conflict_pairs]],
            dtype=torch.long,
        )
    else:
        data.edge_conflict_index = torch.empty((2, 0), dtype=torch.long)

    return data


# ── tests ─────────────────────────────────────────────────────────────────────


def test_correct_increment_on_target_edge() -> None:
    """Stepping on a correct edge yields +1 reward and updates bridge counts."""
    puzzle = _make_puzzle(
        node_capacities=[2, 2],
        fwd_edges=[(0, 1)],
        targets=[2],
    )
    env = HashiEnv(RLConfig(), _DEFAULT_MODEL_CFG)
    _, info = env.reset(puzzle)

    assert info["num_puzzle_edges"] == 1
    assert info["max_steps"] == 2

    obs2, reward, done, step_info = env.step(0)

    assert reward == pytest.approx(1.0)
    assert not done
    assert step_info["terminal_reason"] is None
    assert env.current_bridges is not None
    assert env.current_bridges[0].item() == pytest.approx(1.0)
    # obs edge_attr column 3 reflects the updated count
    assert obs2.edge_attr[0, 3].item() == pytest.approx(1.0)


def test_zero_label_edge_triggers_oracle_failure() -> None:
    """Selecting an edge whose target is 0 immediately triggers oracle_failure."""
    puzzle = _make_puzzle(
        node_capacities=[2, 2],
        fwd_edges=[(0, 1)],
        targets=[0],
    )
    env = HashiEnv(RLConfig(), _DEFAULT_MODEL_CFG)
    env.reset(puzzle)

    # target is 0, so current == target at episode start → oracle failure
    _, reward, done, info = env.step(0)

    assert done
    assert info["terminal_reason"] == "oracle_failure"
    assert reward == pytest.approx(-10.0)


def test_second_increment_on_one_label_edge_triggers_oracle_failure() -> None:
    """Trying to place a second bridge on a target=1 edge triggers oracle_failure."""
    # Two-edge puzzle so the first step does not solve the puzzle.
    puzzle = _make_puzzle(
        node_capacities=[1, 3, 2],
        fwd_edges=[(0, 1), (1, 2)],
        targets=[1, 2],
    )
    env = HashiEnv(RLConfig(), _DEFAULT_MODEL_CFG)
    env.reset(puzzle)

    # First bridge on edge 0 (target=1) — should succeed
    _, reward1, done1, _ = env.step(0)
    assert not done1
    assert reward1 == pytest.approx(1.0)

    # Second bridge on edge 0 (now current == target == 1) → oracle failure
    _, reward2, done2, info2 = env.step(0)
    assert done2
    assert info2["terminal_reason"] == "oracle_failure"
    assert reward2 == pytest.approx(-10.0)


def test_capacity_failure_when_masking_disabled() -> None:
    """Incrementing beyond a node's capacity terminates with capacity_failure."""
    # Node 0 has capacity 1; target=2 is intentionally inconsistent so the
    # oracle allows the second increment while the capacity check fires.
    puzzle = _make_puzzle(
        node_capacities=[1, 3],
        fwd_edges=[(0, 1)],
        targets=[2],
    )
    env = HashiEnv(RLConfig(mask_capacity=False), _DEFAULT_MODEL_CFG)
    env.reset(puzzle)

    # First bridge: oracle OK, capacity OK (unused_cap[0] = 1 > 0)
    _, r1, done1, _ = env.step(0)
    assert not done1
    assert r1 == pytest.approx(1.0)

    # Second bridge: oracle OK (current=1 < target=2), but node 0 is full
    _, r2, done2, info2 = env.step(0)
    assert done2
    assert info2["terminal_reason"] == "capacity_failure"
    assert r2 == pytest.approx(-10.0)


def test_crossing_failure_when_masking_disabled() -> None:
    """A bridge crossing an occupied conflicting edge triggers crossing_failure."""
    # 4 nodes; edges 0 (0→3) and 1 (1→2) cross each other.
    puzzle = _make_puzzle(
        node_capacities=[1, 1, 1, 1],
        fwd_edges=[(0, 3), (1, 2)],
        targets=[1, 1],
        conflict_pairs=[(0, 1)],
    )
    env = HashiEnv(RLConfig(mask_crossing=False), _DEFAULT_MODEL_CFG)
    env.reset(puzzle)

    # Place bridge on edge 0
    _, _, done1, _ = env.step(0)
    assert not done1

    # Place bridge on edge 1 — crosses edge 0 → crossing_failure
    _, reward, done2, info = env.step(1)
    assert done2
    assert info["terminal_reason"] == "crossing_failure"
    assert reward == pytest.approx(-10.0)


def test_capacity_masking_blocks_illegal_action() -> None:
    """With mask_capacity=True, a capacity-violating action is not in the legal mask."""
    puzzle = _make_puzzle(
        node_capacities=[1, 3],
        fwd_edges=[(0, 1)],
        targets=[2],
    )
    env = HashiEnv(RLConfig(mask_capacity=True), _DEFAULT_MODEL_CFG)
    env.reset(puzzle)

    # First bridge exhausts node 0's capacity
    env.step(0)

    mask = env.legal_action_mask()
    # Edge 0 should now be masked because node 0 has no remaining capacity
    assert not mask[0].item()


def test_crossing_masking_blocks_illegal_action() -> None:
    """With mask_crossing=True, a crossing action is absent from the legal mask."""
    puzzle = _make_puzzle(
        node_capacities=[1, 1, 1, 1],
        fwd_edges=[(0, 3), (1, 2)],
        targets=[1, 1],
        conflict_pairs=[(0, 1)],
    )
    env = HashiEnv(RLConfig(mask_crossing=True), _DEFAULT_MODEL_CFG)
    env.reset(puzzle)

    # Place bridge on edge 0 — now edge 1 is a crossing action
    env.step(0)

    mask = env.legal_action_mask()
    # Edge 1 must be masked
    assert not mask[1].item()


def test_solved_detection_on_final_step() -> None:
    """The env reports solved=True with the correct reward on the final bridge."""
    puzzle = _make_puzzle(
        node_capacities=[1, 1],
        fwd_edges=[(0, 1)],
        targets=[1],
    )
    env = HashiEnv(RLConfig(), _DEFAULT_MODEL_CFG)
    env.reset(puzzle)

    _, reward, done, info = env.step(0)

    assert done
    assert info["terminal_reason"] == "solved"
    # reward = reward_correct + reward_solve = 1.0 + 10.0 = 11.0
    assert reward == pytest.approx(11.0)


def test_bidirectional_sync_after_every_step() -> None:
    """current_bridges[j] == current_bridges[j+M] after every step."""
    puzzle = _make_puzzle(
        node_capacities=[2, 2],
        fwd_edges=[(0, 1)],
        targets=[2],
    )
    env = HashiEnv(RLConfig(), _DEFAULT_MODEL_CFG)
    env.reset(puzzle)

    m = env.M  # = 1

    # Step 1
    env.step(0)
    cb = env.current_bridges
    assert cb is not None
    assert cb[0].item() == cb[0 + m].item()

    # Step 2 — puzzle is now solved (target=2, current→2)
    env.step(0)
    cb = env.current_bridges
    assert cb is not None
    assert cb[0].item() == cb[0 + m].item()
    assert cb[0].item() == pytest.approx(2.0)


def test_reset_reinitialises_state() -> None:
    """Calling reset() returns the env to a clean state for a new episode."""
    puzzle = _make_puzzle(
        node_capacities=[1, 1],
        fwd_edges=[(0, 1)],
        targets=[1],
    )
    env = HashiEnv(RLConfig(), _DEFAULT_MODEL_CFG)
    env.reset(puzzle)
    env.step(0)  # solves puzzle; done=True

    assert env.done

    # reset should clear done and bridge counts
    env.reset(puzzle)
    assert not env.done
    assert env.step_count == 0
    cb = env.current_bridges
    assert cb is not None
    assert cb.sum().item() == pytest.approx(0.0)
