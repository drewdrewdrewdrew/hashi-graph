"""Shared node encoder for GNN models."""
import torch
from torch.nn import Embedding, Linear


class NodeEncoder(torch.nn.Module):
    """
    Factorized node encoder with multiple categorical features for logical grounding.

    Features:
        - Logical Capacity: Target bridge count (1-8 for islands, 9/10 for meta)
        - Structural Degree: Number of potential directions (1-4)
        - Unused Capacity: Dynamic remaining bridges needed (signed residual)
        - Conflict Status: Binary flag for crossing-prone edges (0-1)
        - Closeness Centrality: Continuous centrality measure
        - Articulation Points: Binary flag for graph cut vertices (0-1)
        - Spectral Features: First 3 eigenvectors of graph Laplacian
    """

    def __init__(
            self,
            embedding_dim: int,
            hidden_channels: int,
            use_capacity: bool = True,
            use_structural_degree: bool = True,
            use_structural_degree_nsew: bool = False,
            use_unused_capacity: bool = True,
            use_conflict_status: bool = True,
            use_closeness_centrality: bool = True,
            use_articulation_points: bool = False,
            use_spectral_features: bool = False,
            width_mult: float = 1.0,
            depth_mult: int = 1,
            max_capacity: int = 16,
            max_degree: int = 16,
            max_unused: int = 9,
            max_conflict: int = 2,
            capacity_embedding_dim: int | None = None,
            degree_embedding_dim: int | None = None,
            unused_embedding_dim: int | None = None,
            conflict_embedding_dim: int | None = None,
            closeness_embedding_dim: int | None = None,
            ap_embedding_dim: int | None = None,
            spectral_embedding_dim: int | None = None) -> None:
        """
        Initialize NodeEncoder.

        Args:
            embedding_dim (int): Fallback dimensionality of individual feature embeddings.
            hidden_channels (int): Output dimensionality after refinement MLP.
            use_capacity (bool): Whether to embed logical capacity.
            use_structural_degree (bool): Whether to embed structural degree count.
            use_structural_degree_nsew (bool): Whether to embed structural degree as
                NSEW bitmask.
            use_unused_capacity (bool): Whether to embed unused capacity.
            use_conflict_status (bool): Whether to embed conflict status.
            use_closeness_centrality (bool): Whether to include closeness centrality.
            use_articulation_points (bool): Whether to include articulation point.
            use_spectral_features (bool): Whether to include spectral features.
            width_mult (float): Multiplier for hidden dimension in refinement MLP.
            depth_mult (int): Number of hidden layers in refinement MLP.
            max_capacity (int): Max capacity value (exclusive).
            max_degree (int): Max degree value (exclusive).
            max_unused (int): Max unused capacity value (exclusive).
            max_conflict (int): Max conflict status value (exclusive).
            capacity_embedding_dim (int | None): Dimension for capacity embedding.
            degree_embedding_dim (int | None): Dimension for degree embedding.
            unused_embedding_dim (int | None): Dimension for unused capacity embedding.
            conflict_embedding_dim (int | None): Dimension for conflict status embedding.
            closeness_embedding_dim (int | None): Dimension for closeness centrality embedding.
            ap_embedding_dim (int | None): Dimension for articulation point embedding.
            spectral_embedding_dim (int | None): Dimension for spectral feature embedding.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_capacity = use_capacity
        self.use_structural_degree = use_structural_degree
        self.use_structural_degree_nsew = use_structural_degree_nsew
        self.use_unused_capacity = use_unused_capacity
        self.use_conflict_status = use_conflict_status
        self.use_closeness_centrality = use_closeness_centrality
        self.use_articulation_points = use_articulation_points
        self.use_spectral_features = use_spectral_features

        # Set specific dimensions or use fallback
        self.cap_dim = capacity_embedding_dim or embedding_dim
        self.deg_dim = degree_embedding_dim or embedding_dim
        self.un_dim = unused_embedding_dim or embedding_dim
        self.con_dim = conflict_embedding_dim or embedding_dim
        self.close_dim = closeness_embedding_dim or embedding_dim
        self.ap_dim = ap_embedding_dim or embedding_dim
        self.spec_dim = spectral_embedding_dim or embedding_dim

        # Individual feature embeddings
        if use_capacity:
            self.capacity_embedding = Embedding(max_capacity, self.cap_dim)
        if use_structural_degree or use_structural_degree_nsew:
            self.degree_embedding = Embedding(max_degree, self.deg_dim)
        if use_unused_capacity:
            self.unused_embedding = Linear(1, self.un_dim)
        if use_conflict_status:
            self.conflict_embedding = Embedding(max_conflict, self.con_dim)

        # Continuous feature embeddings (Linear layers)
        if use_closeness_centrality:
            self.closeness_embedding = Linear(1, self.close_dim)
        if use_articulation_points:
            self.ap_embedding = Linear(1, self.ap_dim)
        if use_spectral_features:
            self.spectral_embedding = Linear(3, self.spec_dim)  # 3 eigenvectors

        # Refinement MLP to combine factors
        total_input_dim = 0
        if use_capacity:
            total_input_dim += self.cap_dim
        if use_structural_degree or use_structural_degree_nsew:
            total_input_dim += self.deg_dim
        if use_unused_capacity:
            total_input_dim += self.un_dim
        if use_conflict_status:
            total_input_dim += self.con_dim
        if use_closeness_centrality:
            total_input_dim += self.close_dim
        if use_articulation_points:
            total_input_dim += self.ap_dim
        if use_spectral_features:
            total_input_dim += self.spec_dim

        if total_input_dim > 0:
            mlp_layers = []
            curr_dim = total_input_dim
            hidden_dim = int(round(hidden_channels * width_mult))

            # Depth loop: Add intermediate hidden layers
            for _ in range(depth_mult):
                mlp_layers.append(Linear(curr_dim, hidden_dim))
                mlp_layers.append(torch.nn.LayerNorm(hidden_dim))
                mlp_layers.append(torch.nn.ReLU())
                curr_dim = hidden_dim

            # Final projection to hidden_channels
            mlp_layers.append(Linear(curr_dim, hidden_channels))
            self.refiner = torch.nn.Sequential(*mlp_layers)
        else:
            self.refiner = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for node encoding.

        Args:
            x (torch.Tensor): Node features tensor. Expected columns depend on
                enabled features:
                - Column 0: Logical Capacity (if use_capacity)
                - Column 1: Structural Degree (if use_structural_degree or
                  use_structural_degree_nsew)
                - Column 2: Unused Capacity (if use_unused_capacity)
                - Column 3: Conflict Status (if use_conflict_status)
                - Column 4: Closeness Centrality (if use_closeness_centrality)
                - Column 5: Articulation Points (if use_articulation_points)
                - Column 6-8: Spectral Features (if use_spectral_features)

        Returns
        -------
            torch.Tensor of shape [num_nodes, hidden_channels]
        """
        features = []
        col_idx = 0

        # Logical Capacity embedding
        if self.use_capacity:
            capacity_values = x[:, col_idx].long()
            features.append(self.capacity_embedding(capacity_values))
            col_idx += 1

        # Structural Degree embedding (count or NSEW bitmask)
        if self.use_structural_degree or self.use_structural_degree_nsew:
            degree_values = x[:, col_idx].long()
            features.append(self.degree_embedding(degree_values))
            col_idx += 1

        # Unused Capacity (continuous)
        if self.use_unused_capacity:
            unused_values = x[:, col_idx:col_idx + 1]
            features.append(self.unused_embedding(unused_values))
            col_idx += 1

        # Conflict Status embedding
        if self.use_conflict_status:
            conflict_values = x[:, col_idx].long()
            features.append(self.conflict_embedding(conflict_values))
            col_idx += 1

        # Closeness Centrality (continuous)
        if self.use_closeness_centrality:
            closeness_values = x[:, col_idx:col_idx + 1]
            features.append(self.closeness_embedding(closeness_values))
            col_idx += 1

        # Articulation Points (continuous/binary)
        if self.use_articulation_points:
            ap_values = x[:, col_idx:col_idx + 1]
            features.append(self.ap_embedding(ap_values))
            col_idx += 1

        # Spectral Features (continuous)
        if self.use_spectral_features:
            # 3 features
            spec_values = x[:, col_idx:col_idx + 3]
            features.append(self.spectral_embedding(spec_values))
            col_idx += 3

        # Concatenate all feature embeddings
        if features:
            combined = torch.cat(features, dim=-1)
            # Apply refinement MLP if configured
            if self.refiner is not None:
                return self.refiner(combined)
            return combined
        # Fallback if no features enabled
        return torch.zeros(x.size(0), self.embedding_dim, device=x.device)
