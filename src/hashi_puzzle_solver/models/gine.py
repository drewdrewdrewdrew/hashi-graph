"""Graph Isomorphism Network (GINE) model with Edge Features for edge classification."""
import torch
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
import torch.nn.functional as func
from torch_geometric.nn import GINEConv

from .node_encoder import NodeEncoder


class GINEEdgeClassifier(torch.nn.Module):
    """An edge classifier using Graph Isomorphism Network with Edge features (GINE)."""

    def __init__(
            self,
            node_embedding_dim: int,
            hidden_channels: int,
            num_layers: int,
            dropout: float = 0.25,
            use_capacity: bool = True,
            use_structural_degree: bool = True,
            use_structural_degree_nsew: bool = False,
            use_unused_capacity: bool = True,
            use_conflict_status: bool = True,
            use_meta_node: bool = False,
            use_row_col_meta: bool = False,
            edge_dim: int = 3,
            use_closeness_centrality: bool = False,
            use_articulation_points: bool = False,
            use_spectral_features: bool = False,
            use_edge_features_in_prediction: bool = False,
            use_component_meta: bool = False,
            **_kwargs: object) -> None:
        """
        Initialize GINEEdgeClassifier.

        Args:
            node_embedding_dim (int): The dimensionality of the node embeddings.
            hidden_channels (int): The number of channels in the hidden layers.
            num_layers (int): The number of GINE layers.
            dropout (float): Dropout probability. Default: 0.25.
            use_capacity (bool): Whether to embed logical capacity.
            use_structural_degree (bool): Whether to embed structural degree count.
            use_structural_degree_nsew (bool): Whether to embed structural degree as
                NSEW bitmask.
            use_unused_capacity (bool): Whether to embed unused capacity.
            use_conflict_status (bool): Whether to embed conflict status.
            use_meta_node (bool): Whether a meta node is used.
            use_row_col_meta (bool): Whether row/col meta nodes are used.
            edge_dim (int): Dimensionality of edge features.
            use_closeness_centrality (bool): Whether to use closeness centrality.
            use_articulation_points (bool): Whether to use articulation points.
            use_spectral_features (bool): Whether to use spectral features.
            use_edge_features_in_prediction (bool): Whether to concatenate edge
                features to prediction head.
            use_component_meta (bool): Whether to use component meta nodes for
                topological prediction head. Default: False.
            **_kwargs: Additional arguments (ignored).
        """
        super().__init__()
        self.use_capacity = use_capacity
        self.use_structural_degree = use_structural_degree
        self.use_structural_degree_nsew = use_structural_degree_nsew
        self.use_unused_capacity = use_unused_capacity
        self.use_conflict_status = use_conflict_status
        self.use_meta_node = use_meta_node
        self.use_row_col_meta = use_row_col_meta
        self.use_component_meta = use_component_meta
        self.use_edge_features_in_prediction = use_edge_features_in_prediction
        self.node_encoder = NodeEncoder(
            embedding_dim=node_embedding_dim,
            hidden_channels=hidden_channels,
            use_capacity=use_capacity,
            use_structural_degree=use_structural_degree,
            use_structural_degree_nsew=use_structural_degree_nsew,
            use_unused_capacity=use_unused_capacity,
            use_conflict_status=use_conflict_status,
            use_closeness_centrality=use_closeness_centrality,
            use_articulation_points=use_articulation_points,
            use_spectral_features=use_spectral_features
        )
        self.dropout = dropout

        # Node encoder outputs hidden_channels after refinement MLP
        encoder_output_dim = hidden_channels

        # Edge attribute dimension: 3 or 5 (with labels as features)
        self.edge_input_dim = edge_dim

        # Projections for edge attributes to match node feature dimensions
        self.edge_lin_in = Linear(self.edge_input_dim, encoder_output_dim)
        self.edge_lin_hidden = Linear(self.edge_input_dim, hidden_channels)

        self.convs = ModuleList()
        self.bns = ModuleList()

        # First layer: encoder_output_dim -> hidden_channels
        mlp1 = Sequential(
            Linear(encoder_output_dim, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINEConv(mlp1))
        self.bns.append(BatchNorm1d(hidden_channels))

        # Subsequent layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 1):
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINEConv(mlp))
            self.bns.append(BatchNorm1d(hidden_channels))

        # Edge prediction MLP
        # It takes concatenated features of two nodes (+ edge attributes if enabled)
        # (+ component metas if enabled)
        edge_mlp_input_dim = 2 * hidden_channels
        if use_edge_features_in_prediction:
            edge_mlp_input_dim += edge_dim
        if use_component_meta:
            edge_mlp_input_dim += 2 * hidden_channels

        num_classes = 2 if _kwargs.get("head_type") == "ar" else 3
        self.edge_mlp = torch.nn.Sequential(
            Linear(edge_mlp_input_dim, hidden_channels),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, num_classes)
        )

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor | None = None,
            **_kwargs: object) -> torch.Tensor:
        """
        Run forward pass for edge classification.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Graph connectivity.
        edge_attr : torch.Tensor, optional
            Edge attributes. If None, zero attributes will be created.
        **_kwargs : Any
            Additional arguments ignored by GINE.

        Returns
        -------
        torch.Tensor
            Logits for each edge.
        """
        # 1. Encode node features
        h = self.node_encoder(x)

        # Ensure edge_attr is present
        if edge_attr is None:
            # Fallback to zeros if not provided (e.g. legacy code)
            edge_attr = torch.zeros((edge_index.size(1), self.edge_input_dim),
                                  device=x.device, dtype=torch.float)

        # 2. Apply GINE layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns, strict=False)):
            # Project edge attributes to match current node feature dimension
            if i == 0:
                edge_emb = self.edge_lin_in(edge_attr)
            else:
                edge_emb = self.edge_lin_hidden(edge_attr)

            h = conv(h, edge_index, edge_attr=edge_emb)
            h = bn(h)
            h = func.relu(h)
            h = func.dropout(h, p=self.dropout, training=self.training)

        # 3. Predict edge labels
        edge_src, edge_dst = edge_index

        if self.use_component_meta:
            # Topological Prediction Head:
            # [Island A, Meta A, Island B, Meta B]
            node_type = x[:, 0].long()
            comp_e_m = (
                (node_type[edge_index[0]] <= 8) & (node_type[edge_index[1]] == 11)
            )
            island_to_comp_meta = torch.zeros(
                h.size(0), dtype=torch.long, device=h.device
            )
            island_to_comp_meta[edge_index[0, comp_e_m]] = edge_index[1, comp_e_m]

            src_h = h[edge_src]
            dst_h = h[edge_dst]
            src_meta_h = h[island_to_comp_meta[edge_src]]
            dst_meta_h = h[island_to_comp_meta[edge_dst]]

            edge_features = torch.cat([src_h, src_meta_h, dst_h, dst_meta_h], dim=-1)
        else:
            # Standard head: [Island A, Island B]
            edge_features = torch.cat([h[edge_src], h[edge_dst]], dim=-1)

        # Optionally concatenate edge attributes
        if self.use_edge_features_in_prediction:
            edge_features = torch.cat([edge_features, edge_attr], dim=-1)

        return self.edge_mlp(edge_features)
