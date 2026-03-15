"""Modular encoders for node and edge features."""

import torch
from torch.nn import Embedding, Linear, ModuleList, ReLU, LayerNorm, Sequential

from .config import ModelConfig
from .features import NodeFeatureManager, EdgeFeatureManager
from .common import build_mlp

# Constraint vocabulary constants: net_capacity clipped to [-7, 7]
_CV_NC_MIN = -7
_CV_NC_MAX = 7
_CV_NC_BINS = _CV_NC_MAX - _CV_NC_MIN + 1  # 15
_CV_VOCAB_SIZE = 4 * _CV_NC_BINS  # 60


def _constraint_vocab_index(degree: torch.Tensor, net_cap: torch.Tensor) -> torch.Tensor:
    """Compute flat embedding index from (degree, net_capacity) pair."""
    d = degree.long().clamp(1, 4)
    n = net_cap.long().clamp(_CV_NC_MIN, _CV_NC_MAX)
    return (d - 1) * _CV_NC_BINS + (n - _CV_NC_MIN)


class NodeEncoder(torch.nn.Module):
    """
    Modular node encoder that uses NodeFeatureManager for feature mapping.
    """

    def __init__(
        self,
        config: ModelConfig,
        feature_manager: NodeFeatureManager,
        max_capacity: int = 16,
        max_degree: int = 16,
        max_conflict: int = 2,
    ):
        super().__init__()
        self.config = config
        self.fm = feature_manager
        self.embedding_dim = config.node_embedding_dim
        self.hidden_channels = config.hidden_channels

        if config.use_constraint_vocab:
            # Validate: individual replaceable embeddings must be disabled
            if any([
                config.use_capacity,
                config.use_structural_degree,
                config.use_structural_degree_nsew,
                config.use_unused_capacity,
            ]):
                raise ValueError(
                    "use_constraint_vocab=True requires use_capacity, "
                    "use_structural_degree, use_structural_degree_nsew, and "
                    "use_unused_capacity to all be False."
                )
            self.constraint_vocab = Embedding(_CV_VOCAB_SIZE, config.constraint_vocab_dim)

        # Individual feature embeddings (skipped when constraint vocab is active)
        if self.fm.has_feature("capacity") and not config.use_constraint_vocab:
            self.capacity_embedding = Embedding(max_capacity, config.capacity_embedding_dim)

        if self.fm.has_feature("structural_degree") and not config.use_constraint_vocab:
            self.degree_embedding = Embedding(max_degree, config.degree_embedding_dim)

        if self.fm.has_feature("unused_capacity") and not config.use_constraint_vocab:
            self.unused_embedding = Linear(1, config.unused_embedding_dim)

        if self.fm.has_feature("conflict_status"):
            self.conflict_embedding = Embedding(max_conflict, config.conflict_embedding_dim)

        if self.fm.has_feature("closeness_centrality"):
            self.closeness_embedding = Linear(1, config.closeness_embedding_dim)

        if self.fm.has_feature("articulation_point"):
            self.ap_embedding = Linear(1, config.ap_embedding_dim)

        if self.fm.has_feature("spectral_1"):
            self.spectral_embedding = Linear(3, config.spectral_embedding_dim)

        # Refinement MLP to combine factors
        total_input_dim = 0
        if config.use_constraint_vocab:
            total_input_dim += config.constraint_vocab_dim
        else:
            if self.fm.has_feature("capacity"): total_input_dim += config.capacity_embedding_dim
            if self.fm.has_feature("structural_degree"): total_input_dim += config.degree_embedding_dim
            if self.fm.has_feature("unused_capacity"): total_input_dim += config.unused_embedding_dim
        if self.fm.has_feature("conflict_status"): total_input_dim += config.conflict_embedding_dim
        if self.fm.has_feature("closeness_centrality"): total_input_dim += config.closeness_embedding_dim
        if self.fm.has_feature("articulation_point"): total_input_dim += config.ap_embedding_dim
        if self.fm.has_feature("spectral_1"): total_input_dim += config.spectral_embedding_dim
        
        if total_input_dim > 0:
            hidden_dim = int(round(self.hidden_channels * config.node_encoder_width_mult))
            self.refiner = build_mlp(
                input_dim=total_input_dim,
                output_dim=self.hidden_channels,
                hidden_dim=hidden_dim,
                num_layers=config.node_encoder_depth_mult,
                use_layer_norm=True,
            )
        else:
            self.refiner = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode node features into hidden representations."""
        features = []

        if self.config.use_constraint_vocab:
            deg_idx = self.fm.get_idx("structural_degree")
            nc_idx = self.fm.get_idx("unused_capacity")
            vocab_idx = _constraint_vocab_index(x[:, deg_idx], x[:, nc_idx])
            features.append(self.constraint_vocab(vocab_idx))
        else:
            if self.fm.has_feature("capacity"):
                idx = self.fm.get_idx("capacity")
                features.append(self.capacity_embedding(x[:, idx].long()))

            if self.fm.has_feature("structural_degree"):
                idx = self.fm.get_idx("structural_degree")
                features.append(self.degree_embedding(x[:, idx].long()))

            if self.fm.has_feature("unused_capacity"):
                idx = self.fm.get_idx("unused_capacity")
                features.append(self.unused_embedding(x[:, idx : idx + 1]))

        if self.fm.has_feature("conflict_status"):
            idx = self.fm.get_idx("conflict_status")
            features.append(self.conflict_embedding(x[:, idx].long()))

        if self.fm.has_feature("closeness_centrality"):
            idx = self.fm.get_idx("closeness_centrality")
            features.append(self.closeness_embedding(x[:, idx : idx + 1]))

        if self.fm.has_feature("articulation_point"):
            idx = self.fm.get_idx("articulation_point")
            features.append(self.ap_embedding(x[:, idx : idx + 1]))

        if self.fm.has_feature("spectral_1"):
            idx = self.fm.get_idx("spectral_1")
            features.append(self.spectral_embedding(x[:, idx : idx + 3]))

        if not features:
            return torch.zeros(x.size(0), self.hidden_channels, device=x.device)

        combined = torch.cat(features, dim=-1)
        if self.refiner is not None:
            return self.refiner(combined)
        return combined


class EdgeEncoder(torch.nn.Module):
    """
    Modular edge encoder that uses EdgeFeatureManager for feature mapping.
    """

    def __init__(
        self,
        config: ModelConfig,
        feature_manager: EdgeFeatureManager,
    ):
        super().__init__()
        self.config = config
        self.fm = feature_manager
        
        if config.use_categorical_edge_types:
            # We use 9 categories as per existing implementation
            self.edge_type_embedding = Embedding(9, config.edge_type_embedding_dim)

        # Projections for continuous features
        # Distance: 2 inputs (inv_dx, inv_dy)
        self.distance_projector = Sequential(
            Linear(2, config.distance_embedding_dim),
            ReLU(),
            LayerNorm(config.distance_embedding_dim),
        )

        if config.use_continuous_edge_labels:
            # Logits: 3 inputs
            self.logit_projector = Sequential(
                Linear(3, config.logit_embedding_dim),
                ReLU(),
                LayerNorm(config.logit_embedding_dim),
            )

    @property
    def output_dim(self) -> int:
        """Calculate the total output dimension of the encoder."""
        dim = 0
        if self.config.use_categorical_edge_types:
            dim += self.config.edge_type_embedding_dim
        
        # Distance projection is always there as inv_dx/dy are base features
        dim += self.config.distance_embedding_dim
        
        if self.config.use_continuous_edge_labels:
            dim += self.config.logit_embedding_dim
            
        # Binary flags and other features from EdgeFeatureManager that aren't distance or logits
        # We need to count which ones are "raw"
        for name in self.fm.edge_map:
            if name in ["inv_dx", "inv_dy", "bridge_logits"]:
                continue
            # Each of these is a single feature
            dim += 1
            
        return dim

    def forward(self, edge_attr: torch.Tensor, edge_type: torch.Tensor | None = None) -> torch.Tensor:
        """Encode edge attributes and types."""
        features = []

        # 1. Categorical Edge Type Embedding
        if self.config.use_categorical_edge_types:
            if edge_type is None:
                # Fallback to puzzle type (0)
                edge_type = torch.zeros(edge_attr.size(0), dtype=torch.long, device=edge_attr.device)
            features.append(self.edge_type_embedding(edge_type))

        # 2. Distance Projection (inv_dx, inv_dy)
        idx_dx = self.fm.get_idx("inv_dx")
        dist_feats = edge_attr[:, idx_dx : idx_dx + 2]
        features.append(self.distance_projector(dist_feats))

        # 3. Logit Projection
        if self.config.use_continuous_edge_labels:
            idx_logits = self.fm.get_idx("bridge_logits")
            logit_feats = edge_attr[:, idx_logits : idx_logits + 3]
            features.append(self.logit_projector(logit_feats))

        # 4. Pass through remaining binary flags/features as-is
        # We collect all names and indices, sort them, and pick those not already handled
        for name, idx in sorted(self.fm.edge_map.items(), key=lambda x: x[1]):
            if name in ["inv_dx", "inv_dy", "bridge_logits"]:
                continue
            # These are single-column features (is_cut_edge, is_potential_crossing, etc.)
            features.append(edge_attr[:, idx : idx + 1])

        return torch.cat(features, dim=-1)
