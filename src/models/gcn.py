"""
Graph Convolutional Network (GCN) model for edge classification.
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout
from torch_geometric.nn import GCNConv

from .node_encoder import NodeEncoder


class GCNEdgeClassifier(torch.nn.Module):
    """
    An edge classifier using Graph Convolutional Networks (GCN).
    """
    def __init__(self, node_embedding_dim, hidden_channels, num_layers, dropout=0.25,
                 use_capacity=True, use_structural_degree=True, use_structural_degree_nsew=False, use_unused_capacity=True,
                 use_conflict_status=True, use_meta_node=False, use_closeness=False):
        """
        Args:
            node_embedding_dim (int): The dimensionality of the node embeddings.
            hidden_channels (int): The number of channels in the hidden GCN layers.
            num_layers (int): The number of GCN layers.
            dropout (float): Dropout probability. Default: 0.25.
            use_capacity (bool): Whether to embed logical capacity. Default: True.
            use_structural_degree (bool): Whether to embed structural degree count. Default: True.
            use_structural_degree_nsew (bool): Whether to embed structural degree as NSEW bitmask. Default: False.
            use_unused_capacity (bool): Whether to embed unused capacity. Default: True.
            use_conflict_status (bool): Whether to embed conflict status. Default: True.
            use_meta_node (bool): Whether a meta node is used. Default: False.
            use_closeness (bool): Whether to use closeness centrality. Default: False.
        """
        super().__init__()
        self.use_capacity = use_capacity
        self.use_structural_degree = use_structural_degree
        self.use_structural_degree_nsew = use_structural_degree_nsew
        self.use_unused_capacity = use_unused_capacity
        self.use_conflict_status = use_conflict_status
        self.use_meta_node = use_meta_node
        self.node_encoder = NodeEncoder(
            embedding_dim=node_embedding_dim,
            hidden_channels=hidden_channels,
            use_capacity=use_capacity,
            use_structural_degree=use_structural_degree,
            use_structural_degree_nsew=use_structural_degree_nsew,
            use_unused_capacity=use_unused_capacity,
            use_conflict_status=use_conflict_status,
            use_closeness=use_closeness
        )
        self.dropout = dropout

        # Node encoder outputs hidden_channels after refinement MLP
        encoder_output_dim = hidden_channels

        self.convs = ModuleList()
        self.dropouts = ModuleList()
        # Input layer
        self.convs.append(GCNConv(encoder_output_dim, hidden_channels))
        self.dropouts.append(Dropout(dropout))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.dropouts.append(Dropout(dropout))

        # Edge prediction MLP
        # It takes concatenated features of two nodes, hence 2 * hidden_channels
        self.edge_mlp = torch.nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, 3)  # 3 output classes: 0, 1, or 2 bridges
        )

    def forward(self, x, edge_index, **kwargs):
        """
        Forward pass for edge classification.

        Args:
            x (Tensor): Node features of shape [num_nodes, 1] or [num_nodes, 2] if use_degree=True.
            edge_index (LongTensor): Graph connectivity in COO format with
                                     shape [2, num_edges].
            **kwargs: Additional arguments (e.g. batch, edge_attr) ignored by GCN.

        Returns:
            Tensor: Logits for each edge with shape [num_edges, 3].
        """
        # 1. Encode node features
        h = self.node_encoder(x)

        # 2. Apply GCN layers with dropout
        for conv, dropout in zip(self.convs, self.dropouts):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = dropout(h)

        # 3. Predict edge labels
        edge_src, edge_dst = edge_index
        # Concatenate node embeddings for each edge
        edge_features = torch.cat([h[edge_src], h[edge_dst]], dim=-1)

        return self.edge_mlp(edge_features)

