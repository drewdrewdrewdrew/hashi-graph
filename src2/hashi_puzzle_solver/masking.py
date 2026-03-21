"""Masking strategies for curriculum learning in Hashi GNN."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from .models.config import ModelConfig
from .models.features import EdgeFeatureManager, edge_label_column_indices


class MaskingStrategy:
    """Encapsulates progressive masking logic for curriculum learning."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.masking_config = config.get("training", {}).get("masking", {})
        self.enabled = self.masking_config.get("enabled", False)

    def get_rate(self, epoch: int, total_epochs: int) -> float:
        """Calculate progressive masking rate based on epoch."""
        if not self.enabled:
            return 0.0

        warmup_epochs = self.masking_config.get("warmup_epochs", 0)
        cooldown_epochs = self.masking_config.get("cooldown_epochs", 0)
        start_rate = self.masking_config.get("start_rate", 0.0)
        end_rate = self.masking_config.get("end_rate", 1.0)
        schedule = self.masking_config.get("schedule", "cosine")

        rampup_epochs = total_epochs - warmup_epochs - cooldown_epochs
        if rampup_epochs <= 0:
            return start_rate if epoch <= warmup_epochs else end_rate

        if epoch <= warmup_epochs:
            return start_rate
        if epoch > (warmup_epochs + rampup_epochs):
            return end_rate

        progress = (epoch - warmup_epochs) / rampup_epochs
        progress = min(progress, 1.0)

        if schedule == "cosine":
            rate = (
                start_rate
                + (end_rate - start_rate) * (1 - np.cos(np.pi * progress)) / 2
            )
        elif schedule == "linear":
            rate = start_rate + (end_rate - start_rate) * progress
        elif schedule == "constant":
            rate = start_rate
        else:
            msg = f"Unknown masking schedule: {schedule}"
            raise ValueError(msg)

        return float(rate)

    def apply(
        self,
        data: Data,
        masking_rate: float,
        device: torch.device,
    ) -> Data:
        """Mask bridge label and is_labeled features for a subset of edges."""
        if data.edge_attr is None or masking_rate < 0.0:
            return data

        edge_dim = data.edge_attr.size(1)
        if edge_dim < 2:
            return data

        model_config = self.config.get("model", {})
        if not model_config.get("use_edge_labels_as_features", False):
            return data

        # Resolve bridge_label / is_labeled column positions via EdgeFeatureManager
        # so indices remain correct regardless of use_categorical_edge_types or other
        # flag combinations (fixes previous hand-rolled drift for categorical types).
        known_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        try:
            mc = ModelConfig(**{k: v for k, v in model_config.items() if k in known_fields})
        except (TypeError, ValueError):
            return data

        label_indices = edge_label_column_indices(mc)
        if label_indices is None:
            return data
        bridge_label_idx, is_labeled_idx = label_indices

        if edge_dim <= is_labeled_idx:
            return data

        use_capacity = model_config.get("use_capacity", True)
        use_structural_degree = model_config.get("use_structural_degree", True)
        use_structural_degree_nsew = model_config.get(
            "use_structural_degree_nsew", False,
        )
        use_unused_capacity = model_config.get("use_unused_capacity", True)

        unused_capacity_idx = 0
        if use_capacity:
            unused_capacity_idx += 1
        if use_structural_degree or use_structural_degree_nsew:
            unused_capacity_idx += 1

        if use_unused_capacity:
            data.x = data.x.clone()
            data.x[:, unused_capacity_idx] = 0.0

        if masking_rate <= 0.0:
            return data

        data.edge_attr = data.edge_attr.clone()
        original_edge_indices = torch.where(data.edge_mask)[0]
        num_to_mask = int(len(original_edge_indices) * masking_rate)

        if num_to_mask > 0:
            perm = torch.randperm(
                len(original_edge_indices), device=device,
            )[:num_to_mask]
            mask_indices = original_edge_indices[perm]

            if use_unused_capacity:
                original_bridge_labels = data.edge_attr[
                    mask_indices, bridge_label_idx,
                ].clone()

            data.edge_attr[mask_indices, bridge_label_idx] = 0.0
            data.edge_attr[mask_indices, is_labeled_idx] = 0.0

            if use_unused_capacity:
                src_nodes = data.edge_index[0, mask_indices]
                dst_nodes = data.edge_index[1, mask_indices]
                data.x[src_nodes, unused_capacity_idx] += original_bridge_labels
                data.x[dst_nodes, unused_capacity_idx] += original_bridge_labels

        return data
