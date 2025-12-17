"""
Graph Transformer model for edge classification.
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout, LayerNorm
from torch_geometric.nn import TransformerConv

from .node_encoder import NodeEncoder


class TransformerEdgeClassifier(torch.nn.Module):
    """
    An edge classifier using Graph Transformer Convolutions (TransformerConv).
    """
    def __init__(self, node_embedding_dim, hidden_channels, num_layers, 
                 heads=4, dropout=0.25, use_degree=False, use_meta_node=False, use_row_col_meta=False, 
                 edge_dim=3, use_closeness=False):
        """
        Args:
            node_embedding_dim (int): The dimensionality of the node embeddings.
            hidden_channels (int): The number of channels in the hidden layers.
            num_layers (int): The number of Transformer layers.
            heads (int): Number of attention heads. Default: 4.
            dropout (float): Dropout probability. Default: 0.25.
            use_degree (bool): Whether to use node degree as an additional feature.
            use_meta_node (bool): Whether a meta node is used.
            use_row_col_meta (bool): Whether row/col meta nodes are used.
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
        
        # Edge attribute dimension: 3, 4, 5, or 6 (depending on features)
        self.edge_dim = edge_dim

        # Calculate encoder output dimension
        encoder_output_dim = node_embedding_dim
        if use_degree:
            encoder_output_dim += node_embedding_dim
        if use_closeness:
            encoder_output_dim += node_embedding_dim

        self.convs = ModuleList()
        # Optional: LayerNorms can help stabilize deep Transformers
        self.norms = ModuleList() 

        # 1. First Layer: Input -> Hidden
        # TransformerConv expects: in_channels, out_channels
        self.convs.append(TransformerConv(
            encoder_output_dim, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout, 
            edge_dim=self.edge_dim,
            concat=True
        ))
        # Output dim is hidden_channels * heads because concat=True
        self.norms.append(LayerNorm(hidden_channels * heads))

        # 2. Hidden Layers
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(
                hidden_channels * heads, 
                hidden_channels, 
                heads=heads, 
                dropout=dropout, 
                edge_dim=self.edge_dim, 
                concat=True
            ))
            self.norms.append(LayerNorm(hidden_channels * heads))

        # 3. Last Layer: Hidden -> Hidden (reduce heads or project)
        # We usually want a compact representation for the edge classifier.
        # Here we map back to `hidden_channels` with heads=1 (concat=False) or heads=1 (concat=True)
        # Let's align with GAT implementation: reduce to hidden_channels
        if num_layers > 1:
            self.convs.append(TransformerConv(
                hidden_channels * heads, 
                hidden_channels, 
                heads=1, 
                dropout=dropout, 
                edge_dim=self.edge_dim, 
                concat=False
            ))
            # No LayerNorm needed strictly before the final MLP, but consistent features help
            self.norms.append(LayerNorm(hidden_channels))

        final_dim = hidden_channels
        if num_layers == 1:
             # Handle single layer case separately if needed, but loop logic handles >1.
             # If num_layers=1, the first block above was the only one.
             final_dim = hidden_channels * heads

        # Edge prediction MLP
        # It takes concatenated features of two nodes
        self.edge_mlp = torch.nn.Sequential(
            Linear(2 * final_dim, hidden_channels),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, 3)  # 3 output classes: 0, 1, or 2 bridges
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass for edge classification.
        """
        # 1. Encode node features
        h = self.node_encoder(x)
        
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), 
                                  device=x.device, dtype=torch.float)

        # 2. Apply Transformer layers
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Optional: Residual connection if dimensions match
            if h_in.shape == h.shape:
                h = h + h_in

        # 3. Predict edge labels
        edge_src, edge_dst = edge_index
        # Concatenate node embeddings for each edge
        edge_features = torch.cat([h[edge_src], h[edge_dst]], dim=-1)

        return self.edge_mlp(edge_features)

