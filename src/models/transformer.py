"""
Graph Transformer model for edge classification.
"""
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout, LayerNorm
from torch_geometric.nn import TransformerConv

from .node_encoder import NodeEncoder


class TransformerEdgeClassifier(torch.nn.Module):
    """
    An edge classifier using Graph Transformer Convolutions (TransformerConv).
    
    Optionally includes a verification head for self-critique learning, which
    classifies the global meta node embedding to predict whether the model's
    edge predictions form a valid solution.
    """
    def __init__(self, node_embedding_dim, hidden_channels, num_layers, 
                 heads=4, dropout=0.25, use_degree=False, use_meta_node=False, use_row_col_meta=False, 
                 edge_dim=3, use_closeness=False, use_verification_head=False):
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
            use_verification_head (bool): Whether to include verification head. Requires use_meta_node=True.
        """
        super().__init__()
        self.use_degree = use_degree
        self.use_meta_node = use_meta_node
        self.use_row_col_meta = use_row_col_meta
        self.use_verification_head = use_verification_head
        
        # Verification head requires meta node
        if use_verification_head and not use_meta_node:
            raise ValueError("Verification head requires use_meta_node=True")
        
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
        # Each enabled feature (n, degree, closeness) is embedded into node_embedding_dim
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
        
        # Verification head: classifies meta node embedding -> P(valid solution)
        # Used for self-critique learning
        if use_verification_head:
            self.verify_mlp = torch.nn.Sequential(
                Linear(final_dim, hidden_channels // 2),
                torch.nn.ReLU(),
                Dropout(dropout),
                Linear(hidden_channels // 2, 1),
            )

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: Optional[torch.Tensor] = None,
        return_verification: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for edge classification.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            return_verification: If True and verification head is enabled,
                                 also return verification logits from meta node
        
        Returns:
            If return_verification=False:
                edge_logits: [num_edges, 3] edge class logits
            If return_verification=True and verification head enabled:
                Tuple of (edge_logits, verify_logits) where verify_logits is [num_meta_nodes, 1]
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
        edge_logits = self.edge_mlp(edge_features)
        
        # 4. Verification head (if enabled and requested)
        if return_verification and self.use_verification_head:
            # Find global meta nodes (n=9 is global meta node type)
            # x[:, 0] contains the 'n' value (node type/capacity)
            meta_mask = (x[:, 0] == 9)
            meta_embeddings = h[meta_mask]  # [num_meta_nodes, hidden_channels]
            
            verify_logits = self.verify_mlp(meta_embeddings)  # [num_meta_nodes, 1]
            return edge_logits, verify_logits
        
        return edge_logits

