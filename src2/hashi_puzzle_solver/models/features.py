"""Feature orchestration for Hashi Puzzle Solver."""

from typing import Any

from .config import ModelConfig


class NodeFeatureManager:
    """Manages node feature indices and mapping based on ModelConfig."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.node_map = {}
        self.num_node_feats = self._build_schema()

    def _build_schema(self) -> int:
        """Compute the feature schema based on the model configuration."""
        current_node_idx = 0

        if self.config.use_capacity:
            self.node_map["capacity"] = current_node_idx
            current_node_idx += 1
        if (
            self.config.use_structural_degree
            or self.config.use_structural_degree_nsew
        ):
            self.node_map["structural_degree"] = current_node_idx
            current_node_idx += 1
        if self.config.use_unused_capacity:
            self.node_map["unused_capacity"] = current_node_idx
            current_node_idx += 1
        if self.config.use_conflict_status:
            self.node_map["conflict_status"] = current_node_idx
            current_node_idx += 1
        if self.config.use_closeness_centrality:
            self.node_map["closeness_centrality"] = current_node_idx
            current_node_idx += 1
        if self.config.use_articulation_points:
            self.node_map["articulation_point"] = current_node_idx
            current_node_idx += 1
        if self.config.use_spectral_features:
            self.node_map["spectral_1"] = current_node_idx
            self.node_map["spectral_2"] = current_node_idx + 1
            self.node_map["spectral_3"] = current_node_idx + 2
            current_node_idx += 3

        return current_node_idx

    def get_idx(self, name: str) -> int:
        """Get the index of a node feature by name."""
        if name not in self.node_map:
            available = list(self.node_map.keys())
            msg = f"Node feature '{name}' not found. Available: {available}"
            raise ValueError(msg)
        return self.node_map[name]

    def has_feature(self, name: str) -> bool:
        """Check if a feature is enabled."""
        return name in self.node_map


class EdgeFeatureManager:
    """Manages edge feature indices and mapping based on ModelConfig."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.edge_map = {}
        self.num_edge_feats = self._build_schema()

    def _build_schema(self) -> int:
        """Compute the feature schema based on the model configuration."""
        current_edge_idx = 0

        # Base: inv_dx, inv_dy (Always present in dataset but might be zeroed)
        self.edge_map["inv_dx"] = current_edge_idx
        self.edge_map["inv_dy"] = current_edge_idx + 1
        current_edge_idx += 2

        # Conditional flags
        if not self.config.use_categorical_edge_types:
            self.edge_map["is_meta"] = current_edge_idx
            current_edge_idx += 1

            if self.config.use_component_meta:
                self.edge_map["is_comp_membership"] = current_edge_idx
                self.edge_map["is_comp_hierarchy"] = current_edge_idx + 1
                current_edge_idx += 2

                if self.config.use_boundary_flag:
                    self.edge_map["is_boundary"] = current_edge_idx
                    current_edge_idx += 1

            if self.config.use_conflict_edges:
                self.edge_map["is_conflict"] = current_edge_idx
                current_edge_idx += 1
            if self.config.use_meta_mesh:
                self.edge_map["is_meta_mesh"] = current_edge_idx
                current_edge_idx += 1
            if self.config.use_meta_row_col_edges:
                self.edge_map["is_meta_row_col_cross"] = current_edge_idx
                current_edge_idx += 1

        if self.config.use_edge_labels_as_features:
            self.edge_map["bridge_label"] = current_edge_idx
            self.edge_map["is_labeled"] = current_edge_idx + 1
            current_edge_idx += 2
        if self.config.use_cut_edges:
            self.edge_map["is_cut_edge"] = current_edge_idx
            current_edge_idx += 1
        if self.config.use_potential_crossing:
            self.edge_map["is_potential_crossing"] = current_edge_idx
            current_edge_idx += 1
        if self.config.use_continuous_edge_labels:
            self.edge_map["bridge_logits"] = current_edge_idx
            current_edge_idx += 3

        return current_edge_idx

    def get_idx(self, name: str) -> int:
        """Get the index of an edge feature by name."""
        if name not in self.edge_map:
            available = list(self.edge_map.keys())
            msg = f"Edge feature '{name}' not found. Available: {available}"
            raise ValueError(msg)
        return self.edge_map[name]

    def has_feature(self, name: str) -> bool:
        """Check if a feature is enabled."""
        return name in self.edge_map
