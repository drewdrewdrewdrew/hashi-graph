"""Load real Hashi puzzles for RL training and build the standalone policy network."""

from __future__ import annotations

from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from hashi_puzzle_solver.data import HashiDataset
from hashi_puzzle_solver.models.config import ModelConfig as _ModelConfig
from hashi_puzzle_solver.models.features import EdgeFeatureManager
from hashi_puzzle_solver.models.transformer import TransformerEdgeClassifier

if TYPE_CHECKING:
    from torch_geometric.data import Data

_MODEL_CONFIG_FIELDS: frozenset[str] = frozenset(
    f.name for f in _dc_fields(_ModelConfig)
)


def _build_edge_feature_manager(model_cfg: dict[str, Any]) -> EdgeFeatureManager:
    """Construct an :class:`EdgeFeatureManager` from a raw model config dict."""
    mc = _ModelConfig(**{k: v for k, v in model_cfg.items() if k in _MODEL_CONFIG_FIELDS})
    return EdgeFeatureManager(mc)


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

    model_cfg = config["model"]
    if not model_cfg.get("use_edge_labels_as_features", False):
        raise ValueError(
            "load_rl_puzzles requires use_edge_labels_as_features=True in "
            "config['model'].  Enable this flag so the dataset includes "
            "bridge_label / is_labeled columns that HashiEnv writes at each step.  "
            "Also set use_edge_labels_as_features: true in rl_sequential.yaml."
        )

    data_config = config["data"]
    root = Path(data_config["root_dir"])

    dataset = HashiDataset(
        root=root,
        split=split,
        size=data_config.get("size"),
        difficulty=data_config.get("difficulty"),
        limit=None,
        use_degree=model_cfg.get("use_degree", False),
        use_meta_node=model_cfg.get("use_global_meta_node", True),
        use_row_col_meta=model_cfg.get("use_row_col_meta", False),
        use_meta_mesh=model_cfg.get("use_meta_mesh", False),
        use_meta_row_col_edges=model_cfg.get("use_meta_row_col_edges", False),
        use_distance=model_cfg.get("use_distance", False),
        use_edge_labels_as_features=True,  # enforced by assertion above
        use_closeness_centrality=model_cfg.get("use_closeness_centrality", False),
        use_conflict_edges=model_cfg.get("use_conflict_edges", False),
        use_capacity=model_cfg.get("use_capacity", True),
        use_structural_degree=model_cfg.get("use_structural_degree", True),
        use_structural_degree_nsew=model_cfg.get("use_structural_degree_nsew", False),
        use_unused_capacity=model_cfg.get("use_unused_capacity", True),
        use_conflict_status=model_cfg.get("use_conflict_status", True),
        use_articulation_points=model_cfg.get("use_articulation_points", False),
        use_cut_edges=model_cfg.get("use_cut_edges", False),
        use_spectral_features=model_cfg.get("use_spectral_features", False),
        use_potential_crossing=model_cfg.get("use_potential_crossing", False),
        use_component_meta=model_cfg.get("use_component_meta", False),
        use_continuous_edge_labels=model_cfg.get("use_continuous_edge_labels", False),
        use_categorical_edge_types=model_cfg.get("use_categorical_edge_types", False),
        use_constraint_vocab=model_cfg.get("use_constraint_vocab", False),
        transform=None,
    )

    indices = _subset_indices_for_split(config, split, len(dataset))
    puzzles: list[Data] = []
    for idx in indices:
        # No extra column append: bridge_label column already present in
        # edge_attr via use_edge_labels_as_features=True; HashiEnv.reset zeros
        # it and get_obs writes the live count there at each step.
        puzzles.append(dataset[idx].to(device))
    return puzzles


def build_rl_model(
    config: dict[str, Any],
    edge_attr_dim: int,
    device: torch.device,
) -> TransformerEdgeClassifier:
    """Build a ``TransformerEdgeClassifier`` policy from ``config['model']``.

    Sets ``use_rl_edge_encoder=True`` and ``rl_raw_edge_input_dim=edge_attr_dim``.
    When ``use_edge_labels_as_features=True``, also passes an
    ``EdgeFeatureManager`` so the model uses the schema-driven
    ``SchemaRLEdgeEncoder`` instead of the legacy last-column encoder.
    If ``edge_dim`` is omitted, falls back to ``logit_embedding_dim`` (then 32).
    """
    model_cfg = config["model"]
    mc = dict(model_cfg)
    mc.pop("type", None)
    if "use_global_meta_node" in mc:
        mc["use_meta_node"] = mc.pop("use_global_meta_node")
    mc["use_rl_edge_encoder"] = True
    mc["rl_raw_edge_input_dim"] = edge_attr_dim
    if "edge_dim" not in mc:
        mc["edge_dim"] = int(mc.get("logit_embedding_dim", 32))

    # Wire SchemaRLEdgeEncoder when the dataset uses the aligned label schema.
    if model_cfg.get("use_edge_labels_as_features", False):
        mc["rl_edge_feature_manager"] = _build_edge_feature_manager(model_cfg)

    model = TransformerEdgeClassifier(**mc)
    return model.to(device)
