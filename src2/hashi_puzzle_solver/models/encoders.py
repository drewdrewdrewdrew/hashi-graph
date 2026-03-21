"""Modular encoders for node and edge features."""

import torch
from torch.nn import Embedding, LayerNorm, Linear, ModuleList, ReLU, Sequential

from .common import build_mlp
from .config import ModelConfig
from .features import EdgeFeatureManager, NodeFeatureManager

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

    When ``config.use_aligned_label_encoding=True`` (requires
    ``use_edge_labels_as_features=True``), the ``bridge_label`` (3-class) and
    ``is_labeled`` (2-class) columns are encoded with discrete ``Embedding``
    layers instead of being passed through as raw floats.  This mirrors the
    ``SchemaRLEdgeEncoder`` design so that diffusion/iterative models can share
    the same edge representation with the RL policy encoder.  The flag defaults
    to ``False`` — no existing runs are affected.
    """

    def __init__(
        self,
        config: ModelConfig,
        feature_manager: EdgeFeatureManager,
    ):
        super().__init__()
        self.config = config
        self.fm = feature_manager

        if config.use_aligned_label_encoding:
            if not config.use_edge_labels_as_features:
                raise ValueError(
                    "use_aligned_label_encoding=True requires "
                    "use_edge_labels_as_features=True."
                )
            self.bridge_label_embedding = Embedding(3, config.bridge_label_embedding_dim)
            self.is_labeled_embedding = Embedding(2, config.is_labeled_embedding_dim)

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

        # Features from EdgeFeatureManager that aren't distance or logits
        _always_skip = {"inv_dx", "inv_dy", "bridge_logits"}
        for name in self.fm.edge_map:
            if name in _always_skip:
                continue
            if self.config.use_aligned_label_encoding and name in {"bridge_label", "is_labeled"}:
                # Handled via embeddings below
                continue
            dim += 1

        if self.config.use_aligned_label_encoding:
            dim += self.config.bridge_label_embedding_dim
            dim += self.config.is_labeled_embedding_dim

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

        # 4. Remaining single-column features (raw passthrough)
        _skip_raw = {"inv_dx", "inv_dy", "bridge_logits"}
        if self.config.use_aligned_label_encoding:
            _skip_raw = _skip_raw | {"bridge_label", "is_labeled"}

        for name, idx in sorted(self.fm.edge_map.items(), key=lambda x: x[1]):
            if name in _skip_raw:
                continue
            features.append(edge_attr[:, idx : idx + 1])

        # 5. Aligned label embeddings (bridge_label and is_labeled)
        if self.config.use_aligned_label_encoding:
            bl_idx = self.fm.get_idx("bridge_label")
            il_idx = self.fm.get_idx("is_labeled")
            features.append(self.bridge_label_embedding(edge_attr[:, bl_idx].long().clamp(0, 2)))
            features.append(self.is_labeled_embedding(edge_attr[:, il_idx].long().clamp(0, 1)))

        return torch.cat(features, dim=-1)


class SchemaRLEdgeEncoder(torch.nn.Module):
    """Schema-driven RL edge encoder for Hashi bridge placement.

    Embeds ``bridge_label`` (3 classes) and ``is_labeled`` (2 classes) with
    discrete ``Embedding`` layers; linearly projects all remaining continuous
    columns; concatenates the three projections and refines to ``output_dim``
    via an MLP.

    Constructed from an :class:`EdgeFeatureManager` so column indices are
    automatically aligned with the dataset schema, ``MaskingStrategy``, and
    ``HashiEnv`` — replacing the ad-hoc "last column = count" convention of
    the legacy :class:`RLEdgeEncoder`.

    Parameters
    ----------
    feature_manager : EdgeFeatureManager
        Pre-built manager for the same :class:`ModelConfig` used to generate
        ``edge_attr``.  Must have ``use_edge_labels_as_features=True``.
    output_dim : int
        Output embedding dimension (equals the ``edge_dim`` expected by the
        downstream ``TransformerConv`` layers).
    embed_dim : int or None
        Internal embedding width for all three projections.  Defaults to
        ``output_dim``.
    """

    def __init__(
        self,
        feature_manager: EdgeFeatureManager,
        output_dim: int = 16,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not feature_manager.has_feature("bridge_label"):
            raise ValueError(
                "SchemaRLEdgeEncoder requires use_edge_labels_as_features=True "
                "in the EdgeFeatureManager config."
            )
        self.bridge_label_idx = feature_manager.get_idx("bridge_label")
        self.is_labeled_idx = feature_manager.get_idx("is_labeled")

        total_cols = feature_manager.num_edge_feats
        continuous_dim = total_cols - 2  # all columns except bridge_label and is_labeled

        _embed_dim = embed_dim if embed_dim is not None else output_dim

        self.bridge_label_embedding = Embedding(3, _embed_dim)
        self.is_labeled_embedding = Embedding(2, _embed_dim)

        self.continuous_dim = continuous_dim
        if continuous_dim > 0:
            self.continuous_projector: Linear | None = Linear(continuous_dim, _embed_dim)
            refiner_input_dim = 3 * _embed_dim
        else:
            self.continuous_projector = None
            refiner_input_dim = 2 * _embed_dim

        self.refiner = build_mlp(
            input_dim=refiner_input_dim,
            output_dim=output_dim,
            hidden_dim=output_dim,
            num_layers=1,
            use_layer_norm=True,
        )

        # Pre-compute continuous column indices (avoids rebuilding the list on every forward)
        skip = {self.bridge_label_idx, self.is_labeled_idx}
        self._cont_cols: list[int] = [i for i in range(total_cols) if i not in skip]

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Encode raw RL edge attributes into a fixed-size representation.

        Parameters
        ----------
        edge_attr : torch.Tensor
            Shape ``[num_edges, total_cols]``.  ``bridge_label_idx`` column
            holds the current bridge count (clamped to ``[0, 2]``);
            ``is_labeled_idx`` column is 0 for RL inputs (always unmasked).

        Returns
        -------
        torch.Tensor
            Shape ``[num_edges, output_dim]``.
        """
        bl = edge_attr[:, self.bridge_label_idx].long().clamp(0, 2)
        il = edge_attr[:, self.is_labeled_idx].long().clamp(0, 1)

        feats = [
            self.bridge_label_embedding(bl),
            self.is_labeled_embedding(il),
        ]

        if self.continuous_projector is not None and self._cont_cols:
            feats.append(self.continuous_projector(edge_attr[:, self._cont_cols]))

        return self.refiner(torch.cat(feats, dim=-1))


class RLEdgeEncoder(torch.nn.Module):
    """RL-specific edge encoder for Hashi bridge placement.

    Handles the mixed continuous/discrete edge features produced by
    ``HashiEnv.get_obs()``.  The last column of ``edge_attr`` is treated as
    the current bridge count (integer in ``{0, 1, 2}``); all preceding columns
    are treated as continuous features.

    Parameters
    ----------
    input_dim : int
        Total number of raw edge attribute columns.  The last column must be
        the current bridge count; the preceding ``input_dim - 1`` columns are
        continuous features (e.g. ``inv_dx``, ``inv_dy``, ``is_meta``).
    output_dim : int
        Output dimensionality (should equal the ``edge_dim`` expected by the
        downstream ``TransformerConv`` layers).
    embed_dim : int or None
        Internal embedding dimension for both the bridge-count embedding and
        the continuous projector.  Defaults to ``output_dim``.
    """

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 16,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        _embed_dim = embed_dim if embed_dim is not None else output_dim
        continuous_dim = input_dim - 1  # all columns except bridge count

        # Discrete embedding for current bridge count in {0, 1, 2}
        self.bridge_count_embedding = Embedding(3, _embed_dim)
        # Linear projection of continuous positional/structural features
        self.continuous_projector = Linear(continuous_dim, _embed_dim)
        # Refiner MLP: concat(continuous_proj, bridge_emb) -> output_dim
        self.refiner = build_mlp(
            input_dim=2 * _embed_dim,
            output_dim=output_dim,
            hidden_dim=output_dim,
            num_layers=1,
            use_layer_norm=True,
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Encode raw RL edge attributes into a fixed-size representation.

        Parameters
        ----------
        edge_attr : torch.Tensor
            Shape ``[num_edges, input_dim]``.  The last column holds the
            current bridge count (clamped to ``[0, 2]``).

        Returns
        -------
        torch.Tensor
            Shape ``[num_edges, output_dim]``.
        """
        continuous = edge_attr[:, :-1]
        bridge_count = edge_attr[:, -1].long().clamp(0, 2)
        cont_emb = self.continuous_projector(continuous)
        bridge_emb = self.bridge_count_embedding(bridge_count)
        combined = torch.cat([cont_emb, bridge_emb], dim=-1)
        return self.refiner(combined)
