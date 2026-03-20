"""Load real Hashi puzzles for RL training and build the standalone policy network."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from hashi_puzzle_solver.data import HashiDataset
from hashi_puzzle_solver.models.transformer import TransformerEdgeClassifier

if TYPE_CHECKING:
    from torch_geometric.data import Data


def _subset_indices_for_split(
    config: dict[str, Any],
    split: str,
    dataset_len: int,
) -> list[int]:
    """Subset sizes aligned with ``BaseTrainer.create_dataloader``."""
    data_config = config["data"]
    legacy_limit = data_config.get("limit")
    split_limit = (
        data_config.get("train_limit") if split == "train"
        else data_config.get("val_limit")
    )
    limit = split_limit if split_limit is not None else legacy_limit

    if limit is None:
        return list(range(dataset_len))

    num_samples = min(int(limit), dataset_len)
    if split == "train":
        train_seed = int(data_config.get("train_sampler_seed", 42))
        generator = torch.Generator().manual_seed(train_seed)
        perm = torch.randperm(dataset_len, generator=generator)
        return perm[:num_samples].tolist()

    val_sampler_seed = int(data_config.get("val_sampler_seed", 42))
    generator = torch.Generator().manual_seed(val_sampler_seed)
    perm = torch.randperm(dataset_len, generator=generator)
    return perm[:num_samples].tolist()


def _append_rl_bridge_channel(data: Data) -> Data:
    """Append a zero column so the last ``edge_attr`` slot is the live bridge count."""
    out = data.clone()
    if out.edge_attr is None:
        msg = "RL loader requires non-null edge_attr on each graph"
        raise ValueError(msg)
    z = torch.zeros(
        (out.edge_attr.size(0), 1),
        dtype=out.edge_attr.dtype,
        device=out.edge_attr.device,
    )
    out.edge_attr = torch.cat([out.edge_attr, z], dim=1)
    return out


def load_rl_puzzles(
    config: dict[str, Any],
    split: str,
    device: torch.device,
) -> list[Data]:
    """
    Load puzzles for RL.

    Builds ``HashiDataset`` from config (same flags as the base trainer's dataset),
    applies split limits, then appends a zero column to ``edge_attr`` for bridge counts.

    Train uses ``data.train_sampler_seed`` (default 42). Val matches the base trainer's
    seeded ``randperm`` subset (same size as ``SubsetRandomSampler``).
    """
    if split not in ("train", "val"):
        msg = f"split must be 'train' or 'val', got {split!r}"
        raise ValueError(msg)

    data_config = config["data"]
    root = Path(data_config["root_dir"])

    model_config = config["model"]
    dataset = HashiDataset(
        root=root,
        split=split,
        size=data_config.get("size"),
        difficulty=data_config.get("difficulty"),
        limit=None,
        use_degree=model_config.get("use_degree", False),
        use_meta_node=model_config.get("use_global_meta_node", True),
        use_row_col_meta=model_config.get("use_row_col_meta", False),
        use_meta_mesh=model_config.get("use_meta_mesh", False),
        use_meta_row_col_edges=model_config.get(
            "use_meta_row_col_edges",
            False,
        ),
        use_distance=model_config.get("use_distance", False),
        use_edge_labels_as_features=model_config.get(
            "use_edge_labels_as_features",
            False,
        ),
        use_closeness_centrality=model_config.get(
            "use_closeness_centrality",
            False,
        ),
        use_conflict_edges=model_config.get("use_conflict_edges", False),
        use_capacity=model_config.get("use_capacity", True),
        use_structural_degree=model_config.get("use_structural_degree", True),
        use_structural_degree_nsew=model_config.get(
            "use_structural_degree_nsew",
            False,
        ),
        use_unused_capacity=model_config.get("use_unused_capacity", True),
        use_conflict_status=model_config.get("use_conflict_status", True),
        use_articulation_points=model_config.get(
            "use_articulation_points",
            False,
        ),
        use_cut_edges=model_config.get("use_cut_edges", False),
        use_spectral_features=model_config.get("use_spectral_features", False),
        use_potential_crossing=model_config.get(
            "use_potential_crossing",
            False,
        ),
        use_component_meta=model_config.get("use_component_meta", False),
        use_continuous_edge_labels=model_config.get(
            "use_continuous_edge_labels",
            False,
        ),
        use_categorical_edge_types=model_config.get(
            "use_categorical_edge_types",
            False,
        ),
        use_constraint_vocab=model_config.get(
            "use_constraint_vocab",
            False,
        ),
        transform=None,
    )

    indices = _subset_indices_for_split(config, split, len(dataset))
    puzzles: list[Data] = []
    for idx in indices:
        g = _append_rl_bridge_channel(dataset[idx])
        puzzles.append(g.to(device))
    return puzzles


def build_rl_model(
    config: dict[str, Any],
    edge_attr_dim: int,
    device: torch.device,
) -> TransformerEdgeClassifier:
    """
    Build a ``TransformerEdgeClassifier`` policy from ``config['model']``.

    Sets ``use_rl_edge_encoder=True`` and ``rl_raw_edge_input_dim=edge_attr_dim``.
    If ``edge_dim`` is omitted, falls back to ``logit_embedding_dim`` (then 32).
    """
    mc = dict(config["model"])
    mc.pop("type", None)
    if "use_global_meta_node" in mc:
        mc["use_meta_node"] = mc.pop("use_global_meta_node")
    mc["use_rl_edge_encoder"] = True
    mc["rl_raw_edge_input_dim"] = edge_attr_dim
    if "edge_dim" not in mc:
        mc["edge_dim"] = int(mc.get("logit_embedding_dim", 32))

    model = TransformerEdgeClassifier(**mc)
    return model.to(device)
