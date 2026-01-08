"""
Graph Transformer model for edge classification.
"""
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Dropout, LayerNorm
from torch_geometric.nn import TransformerConv, global_mean_pool

from .node_encoder import NodeEncoder


class TransformerEdgeClassifier(torch.nn.Module):
    """
    An edge classifier using Graph Transformer Convolutions (TransformerConv).
    
    Optionally includes a verification head for self-critique learning, which
    classifies the global meta node embedding to predict whether the model's
    edge predictions form a valid solution.
    """
    def __init__(self, node_embedding_dim, hidden_channels, num_layers,
                 heads=4, dropout=0.25, use_capacity=True, use_structural_degree=True,
                 use_structural_degree_nsew=False, use_unused_capacity=True, use_conflict_status=True, use_meta_node=False,
                 use_row_col_meta=False, edge_dim=3, use_closeness=False,
                 use_articulation_points=False, use_spectral_features=False,
                 use_verification_head=False, verifier_use_puzzle_nodes=False, verifier_use_row_col_meta_nodes=False,
                 edge_concat_global_meta=False, **kwargs):
        """
        Args:
            node_embedding_dim (int): The dimensionality of the node embeddings.
            hidden_channels (int): The number of channels in the hidden layers.
            num_layers (int): The number of Transformer layers.
            heads (int): Number of attention heads. Default: 4.
            dropout (float): Dropout probability. Default: 0.25.
            use_capacity (bool): Whether to embed logical capacity. Default: True.
            use_structural_degree (bool): Whether to embed structural degree count. Default: True.
            use_structural_degree_nsew (bool): Whether to embed structural degree as NSEW bitmask. Default: False.
            use_unused_capacity (bool): Whether to embed unused capacity. Default: True.
            use_conflict_status (bool): Whether to embed conflict status. Default: True.
            use_meta_node (bool): Whether a meta node is used.
            use_row_col_meta (bool): Whether row/col meta nodes are used.
            edge_dim (int): Dimensionality of edge features. Default: 3.
            use_closeness (bool): Whether to use closeness centrality. Default: False.
            use_articulation_points (bool): Whether to use articulation points features. Default: False.
            use_spectral_features (bool): Whether to use spectral features. Default: False.
            use_verification_head (bool): Whether to include verification head. Requires use_meta_node=True.
            verifier_use_puzzle_nodes (bool): Whether verification head uses pooled puzzle nodes. Default: False.
            verifier_use_row_col_meta_nodes (bool): Whether verification head uses pooled row/col meta nodes. Default: False.
            edge_concat_global_meta (bool): Whether to concatenate global meta node to edge predictions. Requires use_meta_node=True. Default: False.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__()
        self.use_capacity = use_capacity
        self.use_structural_degree = use_structural_degree
        self.use_structural_degree_nsew = use_structural_degree_nsew
        self.use_unused_capacity = use_unused_capacity
        self.use_conflict_status = use_conflict_status
        self.use_meta_node = use_meta_node
        self.use_row_col_meta = use_row_col_meta
        self.use_verification_head = use_verification_head
        self.verifier_use_puzzle_nodes = verifier_use_puzzle_nodes
        self.verifier_use_row_col_meta_nodes = verifier_use_row_col_meta_nodes
        self.edge_concat_global_meta = edge_concat_global_meta
        
        # Verification head requires meta node
        if use_verification_head and not use_meta_node:
            raise ValueError("Verification head requires use_meta_node=True")

        # Edge global meta concatenation requires meta node
        if edge_concat_global_meta and not use_meta_node:
            raise ValueError("edge_concat_global_meta requires use_meta_node=True")
        
        self.node_encoder = NodeEncoder(
            embedding_dim=node_embedding_dim,
            hidden_channels=hidden_channels,
            use_capacity=use_capacity,
            use_structural_degree=use_structural_degree,
            use_structural_degree_nsew=use_structural_degree_nsew,
            use_unused_capacity=use_unused_capacity,
            use_conflict_status=use_conflict_status,
            use_closeness=use_closeness,
            use_articulation_points=use_articulation_points,
            use_spectral_features=use_spectral_features
        )
        self.dropout = dropout

        # Edge attribute dimension: 3, 4, 5, or 6 (depending on features)
        self.edge_dim = edge_dim

        # Node encoder outputs hidden_channels after refinement MLP
        encoder_output_dim = hidden_channels

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
        # It takes concatenated features of two nodes (+ global meta if enabled)
        edge_mlp_input_dim = 2 * final_dim
        if edge_concat_global_meta:
            edge_mlp_input_dim += final_dim  # Add global meta embedding

        self.edge_mlp = torch.nn.Sequential(
            Linear(edge_mlp_input_dim, hidden_channels),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, 3)  # 3 output classes: 0, 1, or 2 bridges
        )
        
        # Verification head: classifies meta node embedding -> P(valid solution)
        # Used for self-critique learning
        if use_verification_head:
            verify_input_dim = final_dim
            if verifier_use_puzzle_nodes:
                verify_input_dim += final_dim  # Add pooled puzzle node embeddings
            if verifier_use_row_col_meta_nodes:
                verify_input_dim += 2 * final_dim  # Add pooled row and col meta node embeddings
            
            self.verify_mlp = torch.nn.Sequential(
                Linear(verify_input_dim, hidden_channels // 2),
                torch.nn.ReLU(),
                Dropout(dropout),
                Linear(hidden_channels // 2, 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        node_type: Optional[torch.Tensor] = None,
        return_verification: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for edge classification.

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch vector [num_nodes] assigning each node to a graph in the batch
            node_type: Optional node type vector [num_nodes] (1-8 puzzle, 9 global meta, 10 row/col meta).
                      If None, falls back to using x[:, 0] for node type identification.
            return_verification: If True and verification head is enabled,
                                 also return verification logits from meta node

        Returns:
            If return_verification=False:
                edge_logits: [num_edges, 3] edge class logits
            If return_verification=True and verification head enabled:
                Tuple of (edge_logits, verify_logits) where verify_logits is [num_meta_nodes, 1]
        """
        # Use node_type if provided, otherwise fall back to x[:, 0] for backward compatibility
        if node_type is None:
            node_type = x[:, 0].long()
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

        # Optionally concatenate global meta node embedding for global context
        if self.edge_concat_global_meta and self.use_meta_node:
            # Find global meta nodes (node_type=9 is global meta)
            global_meta_mask = (node_type == 9)
            global_meta_emb = h[global_meta_mask]  # [num_graphs, hidden_channels]

            # Get batch indices for each edge to index the correct global meta
            if batch is not None:
                edge_batch = batch[edge_src]  # Use source node batch indices
                global_emb_for_edges = global_meta_emb[edge_batch]
            else:
                # Single graph case - broadcast the same global embedding to all edges
                global_emb_for_edges = global_meta_emb.expand(edge_src.size(0), -1)

            edge_features = torch.cat([edge_features, global_emb_for_edges], dim=-1)

        edge_logits = self.edge_mlp(edge_features)
        
        # 4. Verification head (if enabled and requested)
        if return_verification and self.use_verification_head:
            # Find global meta nodes (node_type=9)
            meta_mask = (node_type == 9)
            meta_embeddings = h[meta_mask]  # [num_graphs, hidden_channels]
            
            verify_input = meta_embeddings
            
            if self.verifier_use_puzzle_nodes:
                # Pool puzzle nodes (islands, node_type <= 8)
                puzzle_mask = (node_type <= 8)
                puzzle_h = h[puzzle_mask]
                
                if batch is not None:
                    puzzle_batch = batch[puzzle_mask]
                    pooled_puzzle = global_mean_pool(puzzle_h, puzzle_batch)
                else:
                    # Single graph case (batch is None)
                    if puzzle_h.size(0) > 0:
                        pooled_puzzle = puzzle_h.mean(dim=0, keepdim=True)
                    else:
                        pooled_puzzle = torch.zeros_like(meta_embeddings)
                
                # Ensure pooled_puzzle matches meta_embeddings batch size
                # (In rare cases with empty graphs or filtering issues)
                if pooled_puzzle.size(0) != meta_embeddings.size(0):
                     # If sizes mismatch, it's likely due to empty graphs or batch alignment.
                     # However, PyG batching should keep them aligned if every graph has a meta node.
                     pass

                verify_input = torch.cat([verify_input, pooled_puzzle], dim=-1)

            if self.verifier_use_row_col_meta_nodes:
                # Pool row/col meta nodes (node_type=10)
                meta_mask_extended = (node_type == 10)  # All row/col meta nodes
                meta_extended_h = h[meta_mask_extended]

                if batch is not None:
                    meta_extended_batch = batch[meta_mask_extended]
                    pooled_meta_extended = global_mean_pool(meta_extended_h, meta_extended_batch)
                else:
                    if meta_extended_h.size(0) > 0:
                        pooled_meta_extended = meta_extended_h.mean(dim=0, keepdim=True)
                    else:
                        pooled_meta_extended = torch.zeros_like(meta_embeddings)

                # For simplicity, use the same pooled embedding for both row and col
                # This could be improved by properly distinguishing row vs col meta nodes
                verify_input = torch.cat([verify_input, pooled_meta_extended, pooled_meta_extended], dim=-1)

            verify_logits = self.verify_mlp(verify_input)  # [num_graphs, 1]
            return edge_logits, verify_logits
        
        return edge_logits
