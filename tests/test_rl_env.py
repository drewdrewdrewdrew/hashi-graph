"""Tests for the HashiEnv oracle-aware RL environment (Chunk 1 + Phase 3)."""

from collections import Counter

from hashi_puzzle_solver.models.config import ModelConfig
from hashi_puzzle_solver.models.features import EdgeFeatureManager
from hashi_puzzle_solver.rl.config import RLConfig
from hashi_puzzle_solver.rl.env import HashiEnv, _label_indices_from_cfg
import pytest
import torch
from torch_geometric.data import Data

# ── test fixture helpers ──────────────────────────────────────────────────────

_DEFAULT_MODEL_CFG = {
    "use_capacity": True,
    "use_structural_degree": True,
    "use_unused_capacity": True,
}

# Minimal schema-aligned model config (use_edge_labels_as_features=True)
_SCHEMA_MODEL_CFG = {
    "use_capacity": True,
    "use_structural_degree": True,
    "use_unused_capacity": True,
    "use_edge_labels_as_features": True,
    "use_categorical_edge_types": False,
    "use_conflict_edges": False,
    "use_meta_mesh": False,
    "use_meta_row_col_edges": False,
    "use_cut_edges": False,
    "use_potential_crossing": False,
    "use_continuous_edge_labels": False,
    "use_component_meta": False,
    "use_boundary_flag": False,
    "use_conflict_status": False,
    "use_closeness_centrality": False,
    "use_articulation_points": False,
    "use_spectral_features": False,
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


# ── Phase 3: schema-driven label index helpers ────────────────────────────────


def test_label_indices_returns_none_when_flag_off() -> None:
    """_label_indices_from_cfg returns None when use_edge_labels_as_features is False."""
    assert _label_indices_from_cfg(_DEFAULT_MODEL_CFG) is None
    assert _label_indices_from_cfg({}) is None


def test_label_indices_correct_for_schema_config() -> None:
    """_label_indices_from_cfg returns indices that match EdgeFeatureManager."""
    result = _label_indices_from_cfg(_SCHEMA_MODEL_CFG)
    assert result is not None
    bl_idx, il_idx = result

    mc = ModelConfig(
        use_edge_labels_as_features=True,
        use_categorical_edge_types=False,
        use_conflict_edges=False,
        use_meta_mesh=False,
        use_meta_row_col_edges=False,
        use_cut_edges=False,
        use_potential_crossing=False,
        use_continuous_edge_labels=False,
        use_component_meta=False,
        use_boundary_flag=False,
    )
    fm = EdgeFeatureManager(mc)
    assert bl_idx == fm.get_idx("bridge_label")
    assert il_idx == fm.get_idx("is_labeled")
    assert il_idx == bl_idx + 1


def _make_schema_puzzle(
    node_capacities: list[int],
    fwd_edges: list[tuple[int, int]],
    targets: list[int],
) -> Data:
    """Build a puzzle with edge_attr aligned to the minimal schema config.

    The edge_attr has ``fm.num_edge_feats`` columns; bridge_label and is_labeled
    are placed at the indices reported by ``_SCHEMA_MODEL_CFG``.  bridge_label
    is pre-filled with the target values to test that reset() zeros them.
    """
    mc = ModelConfig(
        use_edge_labels_as_features=True,
        use_categorical_edge_types=False,
        use_conflict_edges=False,
        use_meta_mesh=False,
        use_meta_row_col_edges=False,
        use_cut_edges=False,
        use_potential_crossing=False,
        use_continuous_edge_labels=False,
        use_component_meta=False,
        use_boundary_flag=False,
    )
    fm = EdgeFeatureManager(mc)
    num_cols = fm.num_edge_feats
    bl_idx = fm.get_idx("bridge_label")
    il_idx = fm.get_idx("is_labeled")

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
    x_rows = [[float(c), float(deg.get(i, 0)), float(c)] for i, c in enumerate(node_capacities)]
    x = torch.tensor(x_rows, dtype=torch.float)
    node_type = torch.tensor(node_capacities, dtype=torch.long)

    edge_attr = torch.zeros(2 * m, num_cols, dtype=torch.float)
    # Pre-fill bridge_label with targets (simulates what HashiDataset stores)
    for i, t in enumerate(targets):
        edge_attr[i, bl_idx] = float(t)
        edge_attr[i + m, bl_idx] = float(t)
    # is_labeled = 1 for all edges (as if fully labeled by dataset)
    edge_attr[:, il_idx] = 1.0
    # inv_dx, inv_dy
    edge_attr[:, fm.get_idx("inv_dx")] = 0.5
    edge_attr[:, fm.get_idx("inv_dy")] = 0.5

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mask=edge_mask,
        y=y,
        node_type=node_type,
        edge_conflict_index=torch.empty((2, 0), dtype=torch.long),
    )


def test_schema_reset_zeros_label_columns() -> None:
    """reset() must zero bridge_label and is_labeled to prevent target leakage."""
    bl_idx, il_idx = _label_indices_from_cfg(_SCHEMA_MODEL_CFG)  # type: ignore[misc]

    puzzle = _make_schema_puzzle(
        node_capacities=[2, 2],
        fwd_edges=[(0, 1)],
        targets=[2],
    )
    # Confirm the puzzle starts with non-zero values in label columns
    assert puzzle.edge_attr[:, bl_idx].sum().item() > 0
    assert puzzle.edge_attr[:, il_idx].sum().item() > 0

    env = HashiEnv(RLConfig(), _SCHEMA_MODEL_CFG)
    obs, _ = env.reset(puzzle)

    assert obs.edge_attr[:, bl_idx].sum().item() == pytest.approx(0.0), (
        "bridge_label must be zeroed after reset"
    )
    assert obs.edge_attr[:, il_idx].sum().item() == pytest.approx(0.0), (
        "is_labeled must be zeroed after reset"
    )


def test_schema_get_obs_writes_bridge_count_to_correct_column() -> None:
    """get_obs writes current_bridges to bridge_label_idx, not the last column."""
    bl_idx, il_idx = _label_indices_from_cfg(_SCHEMA_MODEL_CFG)  # type: ignore[misc]

    puzzle = _make_schema_puzzle(
        node_capacities=[2, 2],
        fwd_edges=[(0, 1)],
        targets=[2],
    )
    env = HashiEnv(RLConfig(), _SCHEMA_MODEL_CFG)
    env.reset(puzzle)

    obs_after_step, _, _, _ = env.step(0)

    # bridge_label column should now be 1 (one bridge placed)
    assert obs_after_step.edge_attr[0, bl_idx].item() == pytest.approx(1.0)
    # is_labeled stays 0 for RL observations
    assert obs_after_step.edge_attr[:, il_idx].sum().item() == pytest.approx(0.0)


def test_schema_get_obs_does_not_write_last_column() -> None:
    """In schema mode the bridge count must NOT appear in the final column
    (unless bridge_label_idx happens to be the last column, which it is not
    for the default non-categorical schema layout).
    """
    bl_idx, il_idx = _label_indices_from_cfg(_SCHEMA_MODEL_CFG)  # type: ignore[misc]
    mc = ModelConfig(
        use_edge_labels_as_features=True,
        use_categorical_edge_types=False,
        use_conflict_edges=False,
        use_meta_mesh=False,
        use_meta_row_col_edges=False,
        use_cut_edges=False,
        use_potential_crossing=False,
        use_continuous_edge_labels=False,
        use_component_meta=False,
        use_boundary_flag=False,
    )
    fm = EdgeFeatureManager(mc)
    last_col = fm.num_edge_feats - 1

    # is_labeled should be the last column for this minimal schema
    assert il_idx == last_col, (
        f"is_labeled should be last column; got il_idx={il_idx} last_col={last_col}"
    )
    # bridge_label is one before it
    assert bl_idx == last_col - 1

    puzzle = _make_schema_puzzle(
        node_capacities=[2, 2],
        fwd_edges=[(0, 1)],
        targets=[2],
    )
    env = HashiEnv(RLConfig(), _SCHEMA_MODEL_CFG)
    env.reset(puzzle)
    obs_after_step, _, _, _ = env.step(0)

    # After a step, bridge_label_idx holds 1; is_labeled holds 0.
    assert obs_after_step.edge_attr[0, bl_idx].item() == pytest.approx(1.0)
    assert obs_after_step.edge_attr[0, il_idx].item() == pytest.approx(0.0)


def test_schema_bidirectional_sync_in_get_obs() -> None:
    """Both forward and reverse edges get the same bridge count in schema mode."""
    bl_idx, _ = _label_indices_from_cfg(_SCHEMA_MODEL_CFG)  # type: ignore[misc]

    puzzle = _make_schema_puzzle(
        node_capacities=[2, 2],
        fwd_edges=[(0, 1)],
        targets=[2],
    )
    env = HashiEnv(RLConfig(), _SCHEMA_MODEL_CFG)
    env.reset(puzzle)
    obs, _, _, _ = env.step(0)

    m = env.M  # = 1
    assert obs.edge_attr[0, bl_idx].item() == pytest.approx(1.0)   # fwd
    assert obs.edge_attr[m, bl_idx].item() == pytest.approx(1.0)   # rev


# ── Phase 3: loader assertion ─────────────────────────────────────────────────


def test_load_rl_puzzles_requires_edge_labels_flag() -> None:
    """load_rl_puzzles raises ValueError when use_edge_labels_as_features is False."""
    import pytest as _pytest
    from hashi_puzzle_solver.rl.loader import load_rl_puzzles

    config = {
        "data": {"root_dir": "dataset/"},
        "model": {
            "use_edge_labels_as_features": False,
        },
    }
    with _pytest.raises(ValueError, match="use_edge_labels_as_features"):
        load_rl_puzzles(config, "train", torch.device("cpu"))


def test_build_rl_model_uses_schema_encoder_when_flag_set() -> None:
    """build_rl_model passes rl_edge_feature_manager so SchemaRLEdgeEncoder is selected."""
    from hashi_puzzle_solver.models.encoders import SchemaRLEdgeEncoder
    from hashi_puzzle_solver.rl.loader import build_rl_model

    config = {
        "model": {
            "type": "transformer",
            "node_embedding_dim": 8,
            "hidden_channels": 16,
            "num_layers": 2,
            "heads": 2,
            "dropout": 0.0,
            "edge_dim": 8,
            "use_edge_labels_as_features": True,
            "use_categorical_edge_types": False,
            "use_conflict_edges": False,
            "use_cut_edges": False,
            "use_potential_crossing": False,
            "use_continuous_edge_labels": False,
            "use_component_meta": False,
            "use_meta_mesh": False,
            "use_meta_row_col_edges": False,
            "use_constraint_vocab": False,
        }
    }
    fm = EdgeFeatureManager(
        ModelConfig(
            use_edge_labels_as_features=True,
            use_categorical_edge_types=False,
            use_conflict_edges=False,
            use_cut_edges=False,
            use_potential_crossing=False,
            use_continuous_edge_labels=False,
            use_component_meta=False,
            use_boundary_flag=False,
            use_meta_mesh=False,
            use_meta_row_col_edges=False,
        )
    )
    edge_attr_dim = fm.num_edge_feats

    model = build_rl_model(config, edge_attr_dim, torch.device("cpu"))
    assert isinstance(model.rl_edge_encoder, SchemaRLEdgeEncoder), (
        "build_rl_model should select SchemaRLEdgeEncoder when use_edge_labels_as_features=True"
    )


def test_build_rl_model_uses_legacy_encoder_when_flag_off() -> None:
    """build_rl_model falls back to RLEdgeEncoder when use_edge_labels_as_features is False."""
    from hashi_puzzle_solver.models.encoders import RLEdgeEncoder
    from hashi_puzzle_solver.rl.loader import build_rl_model

    config = {
        "model": {
            "type": "transformer",
            "node_embedding_dim": 8,
            "hidden_channels": 16,
            "num_layers": 2,
            "heads": 2,
            "dropout": 0.0,
            "edge_dim": 8,
            "use_edge_labels_as_features": False,
        }
    }
    model = build_rl_model(config, 4, torch.device("cpu"))
    assert isinstance(model.rl_edge_encoder, RLEdgeEncoder)
