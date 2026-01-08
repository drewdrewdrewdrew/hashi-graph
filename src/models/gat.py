"""
Graph Attention Network (GAT) model for edge classification.
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout
from torch_geometric.nn import GATConv

from .node_encoder import NodeEncoder


class GATEdgeClassifier(torch.nn.Module):
    """
    An edge classifier using Graph Attention Networks (GAT).
    """
    def __init__(self, node_embedding_dim, hidden_channels, num_layers,
                 heads=8, dropout=0.25, use_capacity=True, use_structural_degree=True,
                 use_structural_degree_nsew=False, use_unused_capacity=True, use_conflict_status=True, use_meta_node=False,
                 use_row_col_meta=False, edge_dim=3, use_closeness=False):
        """
        Args:
            node_embedding_dim (int): The dimensionality of the node embeddings.
            hidden_channels (int): The number of channels in the hidden GAT layers.
            num_layers (int): The number of GAT layers.
            heads (int): Number of attention heads. Default: 8.
            dropout (float): Dropout probability. Default: 0.25.
            use_capacity (bool): Whether to embed logical capacity. Default: True.
            use_structural_degree (bool): Whether to embed structural degree count. Default: True.
            use_structural_degree_nsew (bool): Whether to embed structural degree as NSEW bitmask. Default: False.
            use_unused_capacity (bool): Whether to embed unused capacity. Default: True.
            use_conflict_status (bool): Whether to embed conflict status. Default: True.
            use_meta_node (bool): Whether a meta node is used. Default: False.
            use_row_col_meta (bool): Whether row/col meta nodes are used. Default: False.
            edge_dim (int): Dimensionality of edge features. Default: 3.
            use_closeness (bool): Whether to use closeness centrality. Default: False.
        """
        super().__init__()
        self.use_capacity = use_capacity
        self.use_structural_degree = use_structural_degree
        self.use_structural_degree_nsew = use_structural_degree_nsew
        self.use_unused_capacity = use_unused_capacity
        self.use_conflict_status = use_conflict_status
        self.use_meta_node = use_meta_node
        self.use_row_col_meta = use_row_col_meta
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
        self.heads = heads
        
        # Edge attribute dimension: 3, 4, 5, or 6 (depending on features)
        self.edge_dim = edge_dim

        # Node encoder outputs hidden_channels after refinement MLP
        encoder_output_dim = hidden_channels

        self.convs = ModuleList()
        self.dropouts = ModuleList()

        if num_layers == 1:
            # Single layer: use single head to output hidden_channels directly
            self.convs.append(GATConv(encoder_output_dim, hidden_channels, 
                                      heads=1, dropout=dropout, concat=False, edge_dim=self.edge_dim))
            self.dropouts.append(Dropout(dropout))
            final_hidden_dim = hidden_channels
        else:
            # Input layer: use multiple heads
            self.convs.append(GATConv(encoder_output_dim, hidden_channels, 
                                      heads=heads, dropout=dropout, concat=True, edge_dim=self.edge_dim))
            self.dropouts.append(Dropout(dropout))
            
            # Hidden layers: use multiple heads
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                         heads=heads, dropout=dropout, concat=True, edge_dim=self.edge_dim))
                self.dropouts.append(Dropout(dropout))
            
            # Output layer: single head to reduce dimensions
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                     heads=1, dropout=dropout, concat=False, edge_dim=self.edge_dim))
            self.dropouts.append(Dropout(dropout))
            final_hidden_dim = hidden_channels

        # Edge prediction MLP
        # It takes concatenated features of two nodes, hence 2 * final_hidden_dim
        self.edge_mlp = torch.nn.Sequential(
            Linear(2 * final_hidden_dim, hidden_channels),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, 3)  # 3 output classes: 0, 1, or 2 bridges
        )

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        """
        Forward pass for edge classification.
        
        Args:
            x (Tensor): Node features of shape [num_nodes, 1] or [num_nodes, 2] if use_degree=True.
            edge_index (LongTensor): Graph connectivity in COO format with
                                     shape [2, num_edges].
            edge_attr (Tensor, optional): Edge attributes.
            **kwargs: Additional arguments (e.g. batch) ignored by GAT.
        
        Returns:
            Tensor: Logits for each edge with shape [num_edges, 3].
        """
        # 1. Encode node features
        h = self.node_encoder(x)
        
        # Ensure edge_attr is present
        if edge_attr is None:
             # Fallback to zeros if not provided
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), 
                                  device=x.device, dtype=torch.float)

        # 2. Apply GAT layers with dropout
        for conv, dropout in zip(self.convs, self.dropouts):
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.elu(h)  # GAT typically uses ELU activation
            h = dropout(h)

        # 3. Predict edge labels
        edge_src, edge_dst = edge_index
        # Concatenate node embeddings for each edge
        edge_features = torch.cat([h[edge_src], h[edge_dst]], dim=-1)

        return self.edge_mlp(edge_features)

