"""Factory for creating GNN models for Hashi puzzle solver."""

from typing import Any

import torch

from .gine import GINEEdgeClassifier
from .transformer import TransformerEdgeClassifier


class ModelFactory:
    """Factory class to create and configure model instances."""

    @staticmethod
    def create_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
        """Create and return the model based on config."""
        model_config = config["model"]
        model_type = model_config.get("type", "gine").lower()

        edge_dim = ModelFactory.compute_edge_dim(config)

        common_kwargs = {
            "node_embedding_dim": model_config["node_embedding_dim"],
            "hidden_channels": model_config["hidden_channels"],
            "num_layers": model_config["num_layers"],
            "dropout": model_config.get("dropout", 0.25),
            "use_capacity": model_config.get("use_capacity", True),
            "use_structural_degree": model_config.get("use_structural_degree", True),
            "use_structural_degree_nsew": model_config.get(
                "use_structural_degree_nsew", False,
            ),
            "use_unused_capacity": model_config.get("use_unused_capacity", True),
            "use_conflict_status": model_config.get("use_conflict_status", True),
            "use_meta_node": model_config.get("use_global_meta_node", True),
            "use_closeness_centrality": model_config.get(
                "use_closeness_centrality", False,
            ),
            "use_articulation_points": model_config.get(
                "use_articulation_points", False,
            ),
            "use_spectral_features": model_config.get("use_spectral_features", False),
            "use_edge_features_in_prediction": model_config.get(
                "use_edge_features_in_prediction", False,
            ),
            "use_component_meta": model_config.get("use_component_meta", False),
            "head_type": config.get("training", {}).get("mode", "one-shot"),
        }

        if model_type == "gine":
            model = GINEEdgeClassifier(
                **common_kwargs,
                use_row_col_meta=model_config.get("use_row_col_meta", False),
                edge_dim=edge_dim,
            )
        elif model_type == "transformer":
            model = TransformerEdgeClassifier(
                **common_kwargs,
                heads=model_config.get("heads", 4),
                use_row_col_meta=model_config.get("use_row_col_meta", False),
                edge_dim=edge_dim,
                use_verification_head=model_config.get("use_verification_head", False),
                verifier_use_puzzle_nodes=model_config.get(
                    "verifier_use_puzzle_nodes", False,
                ),
                verifier_use_row_col_meta_nodes=model_config.get(
                    "verifier_use_row_col_meta_nodes", False,
                ),
                edge_concat_global_meta=model_config.get(
                    "edge_concat_global_meta", False,
                ),
            )
        else:
            msg = f"Unknown model type: {model_type}"
            raise ValueError(msg)

        return model.to(device)

    @staticmethod
    def compute_edge_dim(config: dict[str, Any]) -> int:
        """Calculate edge dimension based on enabled features."""
        model_config = config["model"]
        edge_dim = 3  # base: [inv_dx, inv_dy, is_meta]
        if model_config.get("use_conflict_edges", False):
            edge_dim += 1
        if model_config.get("use_meta_mesh", False):
            edge_dim += 1
        if model_config.get("use_meta_row_col_edges", False):
            edge_dim += 1
        if model_config.get("use_edge_labels_as_features", False):
            edge_dim += 2
        if model_config.get("use_cut_edges", False):
            edge_dim += 1
        return edge_dim
