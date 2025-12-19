"""
Shared node encoder for GNN models.
"""
import torch
from torch.nn import Embedding, Linear


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
            embedding_dim (int): The dimensionality of the island type embedding.
            max_n_value (int): The maximum possible 'n' value (exclusive).
            Default 11 covers: 0 (unused), 1-8 (puzzle), 9 (global meta), 10 (line meta).
            use_degree (bool): Whether to use node degree as a feature.
            use_meta_node (bool): Whether a meta node is used.
            use_row_col_meta (bool): Whether row/col meta nodes are used.
            use_closeness (bool): Whether closeness centrality is included as a feature.
        """
        super().__init__()
        self.use_degree = use_degree
        self.use_meta_node = use_meta_node
        self.use_row_col_meta = use_row_col_meta
        self.use_closeness = use_closeness
        
        # Embedding for categorical node type (capacity + meta types)
        self.embedding = Embedding(max_n_value, embedding_dim)
        
        # Degree is categorical (1-4 for islands, higher for meta-nodes)
        if use_degree:
            # 500 is a safe upper bound for potential degree in large puzzles
            self.degree_embedding = Embedding(500, embedding_dim)
        
        # Closeness is continuous and scales with board size -> Linear is safer
        if use_closeness:
            self.closeness_embedding = Linear(1, embedding_dim)

    def forward(self, x):
        """
        Forward pass for node encoding.
        
        Returns:
            Tensor of shape [num_nodes, dim]
        """
        # Extract discrete feature (n)
        n_values = x[:, 0].long()
        n_emb = self.embedding(n_values)
        
        features = [n_emb]
        
        col_idx = 1
        if self.use_degree:
            # Degree indices (integers)
            degree_indices = x[:, col_idx].long().clamp(0, 499)
            features.append(self.degree_embedding(degree_indices))
            col_idx += 1
            
        if self.use_closeness:
            # Closeness scalar
            closeness_values = x[:, col_idx:col_idx+1]
            features.append(self.closeness_embedding(closeness_values))
        
        # Concatenate embeddings
        return torch.cat(features, dim=-1)

