#!/usr/bin/env python3
"""Minimal REINFORCE smoke test using configs/rl_sequential.yaml (synthetic puzzles)."""

from __future__ import annotations

import pathlib
from dataclasses import fields

import torch
import yaml
from torch_geometric.data import Data

_REPO = pathlib.Path(__file__).resolve().parent.parent

from hashi_puzzle_solver.models.transformer import TransformerEdgeClassifier
from hashi_puzzle_solver.rl.config import RLConfig
from hashi_puzzle_solver.rl.reinforce import ReinforceTrainer


def _load_yaml(path: pathlib.Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _rl_config_from_dict(raw: dict) -> RLConfig:
    r = raw.get("rl") or {}
    names = {f.name for f in fields(RLConfig)}
    return RLConfig(**{k: r[k] for k in names if k in r})


def _make_puzzle(
    node_capacities: list[int],
    fwd_edges: list[tuple[int, int]],
    targets: list[int],
) -> Data:
    from collections import Counter

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


def main() -> None:
    cfg_path = _REPO / "configs" / "rl_sequential.yaml"
    raw = _load_yaml(cfg_path)
    rl_cfg = _rl_config_from_dict(raw)
    rt = raw.get("rl_training") or {}
    lr = float(rt.get("policy_learning_rate", 0.0001))
    max_steps = int(rt.get("max_steps_per_rollout", 200))
    n_updates = int(rt.get("smoke_updates", 5))

    puzzles = [
        _make_puzzle([2, 2], [(0, 1)], [1]),
        _make_puzzle([3, 1, 2], [(0, 1), (0, 2)], [1, 1]),
    ]

    torch.manual_seed(0)
    model = TransformerEdgeClassifier(
        node_embedding_dim=16,
        hidden_channels=32,
        num_layers=2,
        heads=2,
        dropout=0.0,
        edge_dim=16,
        use_capacity=True,
        use_structural_degree=True,
        use_unused_capacity=True,
        use_conflict_status=False,
        use_rl_edge_encoder=True,
        rl_raw_edge_input_dim=4,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ReinforceTrainer(model, opt, rl_cfg, puzzles)

    print(f"Loaded {cfg_path.name} — running {n_updates} REINFORCE updates...")
    for i, m in enumerate(trainer.train(n_updates, max_steps=max_steps), start=1):
        print(f"  update {i}: loss={m['loss']:.4f}")
    ev = trainer.evaluate(max_steps=max_steps)
    print("  greedy eval:", {k: round(v, 4) for k, v in ev.items()})


if __name__ == "__main__":
    main()
