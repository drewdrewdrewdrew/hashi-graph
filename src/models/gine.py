"""
Graph Isomorphism Network (GINE) model with Edge Features for edge classification.
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GINEConv

from .node_encoder import NodeEncoder


class GINEEdgeClassifier(torch.nn.Module):
    """
    An edge classifier using Graph Isomorphism Network with Edge features (GINE).
    """
    def __init__(self, node_embedding_dim, hidden_channels, num_layers, 
                 dropout=0.25, use_degree=False, use_meta_node=False, use_row_col_meta=False, 
                 edge_dim=3, use_closeness=False):
        """
        Args:
            node_embedding_dim (int): The dimensionality of the node embeddings.
            hidden_channels (int): The number of channels in the hidden layers.
            num_layers (int): The number of GINE layers.
            dropout (float): Dropout probability. Default: 0.25.
            use_degree (bool): Whether to use node degree as an additional feature. Default: False.
            use_meta_node (bool): Whether a meta node is used. Default: False.
            use_row_col_meta (bool): Whether row/col meta nodes are used. Default: False.
            edge_dim (int): Dimensionality of edge features. Default: 3.
            use_closeness (bool): Whether to use closeness centrality. Default: False.
        """
        super().__init__()
        self.use_degree = use_degree
        self.use_meta_node = use_meta_node
        self.use_row_col_meta = use_row_col_meta
        self.node_encoder = NodeEncoder(
            embedding_dim=node_embedding_dim,
            use_degree=use_degree,
            use_meta_node=use_meta_node,
            use_row_col_meta=use_row_col_meta,
            use_closeness=use_closeness
        )
        self.dropout = dropout

        # Calculate encoder output dimension
        # Each enabled feature (n, degree, closeness) is embedded into node_embedding_dim
        encoder_output_dim = node_embedding_dim
        if use_degree:
            encoder_output_dim += node_embedding_dim
        if use_closeness:
            encoder_output_dim += node_embedding_dim
        
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
        # It takes concatenated features of two nodes: 2 * hidden_channels
        self.edge_mlp = torch.nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, 3)  # 3 output classes: 0, 1, or 2 bridges
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass for edge classification.

        Args:
            x (Tensor): Node features.
            edge_index (LongTensor): Graph connectivity.
            edge_attr (Tensor, optional): Edge attributes. 
                If None, zero attributes will be created (not recommended for GINE).
        """
        # 1. Encode node features
        h = self.node_encoder(x)
        
        # Ensure edge_attr is present
        if edge_attr is None:
            # Fallback to zeros if not provided (e.g. legacy code)
            edge_attr = torch.zeros((edge_index.size(1), self.edge_input_dim), 
                                  device=x.device, dtype=torch.float)

        # 2. Apply GINE layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # Project edge attributes to match current node feature dimension
            if i == 0:
                edge_emb = self.edge_lin_in(edge_attr)
            else:
                edge_emb = self.edge_lin_hidden(edge_attr)
            
            h = conv(h, edge_index, edge_attr=edge_emb)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # 3. Predict edge labels
        edge_src, edge_dst = edge_index
        # Concatenate node embeddings for each edge
        edge_features = torch.cat([h[edge_src], h[edge_dst]], dim=-1)

        return self.edge_mlp(edge_features)
