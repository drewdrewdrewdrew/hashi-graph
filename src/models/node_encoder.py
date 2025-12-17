"""
Shared node encoder for GNN models.
"""
import torch
from torch.nn import Embedding


class NodeEncoder(torch.nn.Module):
    """
    Encodes node features using a learnable embedding.
    The 'n' value encodes both island capacity AND node type:
        - n ∈ {1-8}: Puzzle nodes (island capacity)
        - n = 9: Global meta node
        - n = 10: Line meta node (row or column)
    Optionally supports separate embedding for node degree.
    Optionally supports continuous closeness centrality feature.
    """
    def __init__(self, embedding_dim, max_n_value=11, use_degree=False, use_meta_node=False, 
                 use_row_col_meta=False, use_closeness=False):
        """
        Args:
            embedding_dim (int): The dimensionality of the embedding.
            max_n_value (int): The maximum possible 'n' value (exclusive).
                               Default 11 covers: 0 (unused), 1-8 (puzzle), 9 (global meta), 10 (line meta).
            use_degree (bool): Whether to use node degree as an additional feature. Default: False.
            use_meta_node (bool): Whether a meta node is used. If True and use_degree=True,
                                 increases max_degree to accommodate meta node. Default: False.
            use_row_col_meta (bool): Whether row/col meta nodes are used. Default: False.
            use_closeness (bool): Whether closeness centrality is included as continuous feature. Default: False.
        """
        super().__init__()
        self.use_degree = use_degree
        self.use_meta_node = use_meta_node
        self.use_row_col_meta = use_row_col_meta
        self.use_closeness = use_closeness
        # Embedding for values 0 through 8 (0 is used for meta node).
        self.embedding = Embedding(max_n_value, embedding_dim)
        
        if use_degree:
            # Max degree for Hashi puzzles is 4 (north, south, east, west connections)
            # Meta node uses degree value 5 as a special marker (> 4)
            # Row/Col meta nodes use degree value 6 and 7
            # Note: degree is stored as inverse (1/degree * 100) in data.py
            max_degree = 100  # Inverse degree scaled to 0-100 range
            if use_meta_node:
                max_degree = 100  # Still bounded
            if use_row_col_meta:
                max_degree = 100  # Still bounded
                
            self.degree_embedding = Embedding(max_degree + 1, embedding_dim)
        
        if use_closeness:
            # Closeness is continuous, project it to embedding_dim
            from torch.nn import Linear
            self.closeness_proj = Linear(1, embedding_dim)

    def forward(self, x):
        """
        Forward pass for node encoding.
        
        Args:
            x: Node features tensor. Shape depends on enabled features:
               - [num_nodes, 1] if only 'n' values
               - [num_nodes, 2] if use_degree=True ([n, inv_degree])
               - [num_nodes, 2] if use_closeness=True ([n, closeness])
               - [num_nodes, 3] if both use_degree and use_closeness ([n, inv_degree, closeness])
        
        Returns:
            Tensor of shape [num_nodes, embedding_dim * num_discrete_features + closeness_dim]
        """
        # Extract discrete features (n, degree)
        n_values = x[:, 0].long()
        n_emb = self.embedding(n_values)
        
        embeddings = [n_emb]
        
        col_idx = 1  # Track which column we're on
        
        if self.use_degree:
            # Degree is second column (inverse degree scaled to 0-100)
            degree_values = x[:, col_idx].long()
            degree_emb = self.degree_embedding(degree_values)
            embeddings.append(degree_emb)
            col_idx += 1
        
        # Handle closeness (continuous feature)
        if self.use_closeness:
            # Closeness is the last column (after n, and optionally degree)
            closeness_values = x[:, col_idx:col_idx+1]  # Keep as [N, 1]
            closeness_emb = self.closeness_proj(closeness_values)
            embeddings.append(closeness_emb)
        
        # Concatenate all embeddings
        return torch.cat(embeddings, dim=-1)

