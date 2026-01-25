"""Graph Transformer model for edge classification."""

import torch
from torch.nn import Dropout, LayerNorm, Linear, ModuleList
import torch.nn.functional as func
from torch_geometric.nn import TransformerConv, global_mean_pool

from .node_encoder import NodeEncoder


class TransformerEdgeClassifier(torch.nn.Module):
    """
    An edge classifier using Graph Transformer Convolutions (TransformerConv).

    Optionally includes a verification head for self-critique learning, and a
    sigma head for noise level prediction in continuous diffusion.
    """

    def __init__(
            self,
            node_embedding_dim: int,
            hidden_channels: int,
            num_layers: int,
            heads: int = 4,
            dropout: float = 0.25,
            use_capacity: bool = True,
            use_structural_degree: bool = True,
            use_structural_degree_nsew: bool = False,
            use_unused_capacity: bool = True,
            use_conflict_status: bool = True,
            use_meta_node: bool = False,
            use_row_col_meta: bool = False,
            edge_dim: int = 3,
            use_categorical_edge_types: bool = False,
            use_continuous_edge_labels: bool = False,
            use_closeness_centrality: bool = False,
            use_articulation_points: bool = False,
            use_spectral_features: bool = False,
            use_verification_head: bool = False,
            use_noise_head: bool = False,
            use_time_conditioning: bool = False,
            verifier_use_puzzle_nodes: bool = False,
            verifier_use_row_col_meta_nodes: bool = False,
            edge_concat_global_meta: bool = False,
            use_component_meta: bool = False,
            edge_concat_component_meta: bool = False,
            component_merge_margin: float = 0.5,
            edge_mlp_width_mult: float = 1.0,
            edge_mlp_depth_mult: int = 1,
            node_encoder_width_mult: float = 1.0,
            node_encoder_depth_mult: int = 1,
            noise_mlp_width_mult: float = 0.5,
            noise_mlp_depth_mult: int = 1,
            max_capacity: int = 16,
            max_degree: int = 16,
            max_unused: int = 9,
            max_conflict: int = 2,
            use_edge_features_in_prediction: bool = False,
            **_kwargs: object):
        """
        Initialize the TransformerEdgeClassifier.

        Args:
            node_embedding_dim (int): The dimensionality of the node embeddings.
            hidden_channels (int): The number of channels in the hidden layers.
            num_layers (int): The number of Transformer layers.
            heads (int): Number of attention heads. Default: 4.
            dropout (float): Dropout probability. Default: 0.25.
            use_capacity (bool): Whether to embed logical capacity. Default: True.
            use_structural_degree (bool): Whether to embed structural degree count.
                Default: True.
            use_structural_degree_nsew (bool): Whether to embed structural degree as
                NSEW bitmask. Default: False.
            use_unused_capacity (bool): Whether to embed unused capacity. Default: True.
            use_conflict_status (bool): Whether to embed conflict status. Default: True.
            use_meta_node (bool): Whether a meta node is used.
            use_row_col_meta (bool): Whether row/col meta nodes are used.
            edge_dim (int): Dimensionality of edge features. Default: 3.
            use_categorical_edge_types (bool): Whether to use learned embeddings for 
                categorical edge types. Default: False.
            use_continuous_edge_labels (bool): Whether to use continuous edge logits
                instead of discrete embeddings. Default: False.
            use_closeness_centrality (bool): Whether to use closeness centrality.
                Default: False.
            use_articulation_points (bool): Whether to use articulation points features.
                Default: False.
            use_spectral_features (bool): Whether to use spectral features. Default:
                False.
            use_verification_head (bool): Whether to include verification head.
                Requires use_meta_node=True.
            use_noise_head (bool): Whether to include consolidated noise prediction
                head (for continuous diffusion). Requires use_meta_node=True.
            verifier_use_puzzle_nodes (bool): Whether verification head uses pooled
                puzzle nodes. Default: False.
            verifier_use_row_col_meta_nodes (bool): Whether verification head uses
                pooled row/col meta nodes. Default: False.
            edge_concat_global_meta (bool): Whether to concatenate global meta node to
                edge predictions. Requires use_meta_node=True. Default: False.
            use_component_meta (bool): Whether to use component meta nodes for
                topological prediction head. Default: False.
            edge_concat_component_meta (bool): Whether to concatenate component meta 
                embeddings to edge predictions. Default: False.
            max_capacity (int): Max capacity.
            max_degree (int): Max degree.
            max_unused (int): Max unused.
            max_conflict (int): Max conflict.
            use_edge_features_in_prediction (bool): Whether to use edge features.
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
        self.edge_concat_component_meta = edge_concat_component_meta
        self.component_merge_margin = component_merge_margin
        self.use_categorical_edge_types = use_categorical_edge_types
        self.use_continuous_edge_labels = use_continuous_edge_labels
        self.use_verification_head = use_verification_head
        self.use_noise_head = use_noise_head
        self.use_time_conditioning = use_time_conditioning
        self.verifier_use_puzzle_nodes = verifier_use_puzzle_nodes
        self.verifier_use_row_col_meta_nodes = verifier_use_row_col_meta_nodes
        self.edge_concat_global_meta = edge_concat_global_meta
        self.use_edge_features_in_prediction = use_edge_features_in_prediction

        # Verification head requires meta node
        if use_verification_head and not use_meta_node:
            msg = "Verification head requires use_meta_node=True"
            raise ValueError(msg)

        # Noise head requires meta node
        if use_noise_head and not use_meta_node:
            msg = "Noise head requires use_meta_node=True"
            raise ValueError(msg)

        # Edge global meta concatenation requires meta node
        if edge_concat_global_meta and not use_meta_node:
            msg = "edge_concat_global_meta requires use_meta_node=True"
            raise ValueError(msg)

        # Edge component meta concatenation requires component meta nodes
        if edge_concat_component_meta and not use_component_meta:
            msg = "edge_concat_component_meta requires use_component_meta=True"
            raise ValueError(msg)

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
            use_spectral_features=use_spectral_features,
            width_mult=node_encoder_width_mult,
            depth_mult=node_encoder_depth_mult,
            max_capacity=max_capacity,
            max_degree=max_degree,
            max_unused=max_unused,
            max_conflict=max_conflict
        )
        self.dropout = dropout

        if use_categorical_edge_types:
            self.edge_type_embedding = torch.nn.Embedding(9, node_embedding_dim)

        # Edge attribute dimension
        self.edge_dim = edge_dim

        # Node encoder outputs hidden_channels after refinement MLP
        encoder_output_dim = hidden_channels

        self.convs = ModuleList()
        # Optional: LayerNorms can help stabilize deep Transformers
        self.norms = ModuleList()

        # 1. First Layer: Input -> Hidden
        self.convs.append(TransformerConv(
            encoder_output_dim,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=self.edge_dim,
            concat=True
        ))
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

        # 3. Last Layer: Hidden -> Hidden
        if num_layers > 1:
            self.convs.append(TransformerConv(
                hidden_channels * heads,
                hidden_channels,
                heads=1,
                dropout=dropout,
                edge_dim=self.edge_dim,
                concat=False
            ))
            self.norms.append(LayerNorm(hidden_channels))

        final_dim = hidden_channels
        if num_layers == 1:
            final_dim = hidden_channels * heads

        # Edge prediction MLP
        edge_mlp_input_dim = 2 * final_dim
        if edge_concat_global_meta:
            edge_mlp_input_dim += final_dim
        if edge_concat_component_meta:
            edge_mlp_input_dim += 2 * final_dim
        if use_edge_features_in_prediction:
            edge_mlp_input_dim += self.edge_dim

        # Dynamic MLP construction based on multipliers
        mlp_layers = []
        curr_dim = edge_mlp_input_dim
        hidden_dim = int(round(hidden_channels * edge_mlp_width_mult))

        # Depth loop: Add intermediate hidden layers
        for _ in range(edge_mlp_depth_mult):
            mlp_layers.append(Linear(curr_dim, hidden_dim))
            mlp_layers.append(torch.nn.ReLU())
            mlp_layers.append(Dropout(dropout))
            curr_dim = hidden_dim

        # Final projection to 3 classes
        mlp_layers.append(Linear(curr_dim, 3))
        self.edge_mlp = torch.nn.Sequential(*mlp_layers)

        # Verification head
        if use_verification_head:
            verify_input_dim = final_dim
            if verifier_use_puzzle_nodes:
                verify_input_dim += final_dim
            if verifier_use_row_col_meta_nodes:
                verify_input_dim += 2 * final_dim

            self.verify_mlp = torch.nn.Sequential(
                Linear(verify_input_dim, hidden_channels // 2),
                torch.nn.ReLU(),
                Dropout(dropout),
                Linear(hidden_channels // 2, 1),
            )

        # Consolidates old sigma/alpha heads into a single 'Prophet' head
        # with statistical pooling injection.
        if use_noise_head:
            # Stats Pooling: Mean/Std for
            # [RawLogits(3), Entropy(1), Confidence(1), Margin(1)]
            # Total = (3+1+1+1) * 2 = 12 dimensions
            self.stats_dim = 12
            
            # Dynamic MLP construction based on multipliers
            noise_mlp_layers = []
            curr_dim = final_dim + self.stats_dim
            noise_hidden_dim = int(round(hidden_channels * noise_mlp_width_mult))
            
            for _ in range(noise_mlp_depth_mult):
                noise_mlp_layers.append(Linear(curr_dim, noise_hidden_dim))
                noise_mlp_layers.append(torch.nn.ReLU())
                noise_mlp_layers.append(Dropout(dropout))
                curr_dim = noise_hidden_dim
                
            # Final projection to 2 parameters (sigma, alpha)
            noise_mlp_layers.append(Linear(curr_dim, 2))
            self.diffusion_aux_mlp = torch.nn.Sequential(*noise_mlp_layers)
            
            # Input Noise Embedding (for injection into global meta node)
            self.noise_embedder = Linear(2, final_dim)
        else:
            self.diffusion_aux_mlp = None

        if use_time_conditioning:
            self.time_embedder = torch.nn.Sequential(
                Linear(1, hidden_channels // 4),
                torch.nn.ReLU(),
                Linear(hidden_channels // 4, final_dim)
            )
        else:
            self.time_embedder = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        node_type: torch.Tensor | None = None,
        return_verification: bool = False,
        return_noise: bool = False,
        input_noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        **_kwargs: object
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass for edge classification and optional heads."""
        if node_type is None:
            node_type = x[:, 0].long()

        h = self.node_encoder(x)

        # 0. Noise Injection (Input)
        # Inject input alpha/sigma into Global Meta Node embedding if provided
        if input_noise is not None and self.use_noise_head:
            noise_emb = self.noise_embedder(input_noise)
            global_meta_mask = (node_type == 9)

            # Avoid in-place modification of h to prevent gradient errors
            h_new = h.clone()
            h_new[global_meta_mask] = h_new[global_meta_mask] + noise_emb
            h = h_new

        # 0b. Time Conditioning Injection
        if time is not None and self.use_time_conditioning:
            # time is [num_graphs, 1]
            time_emb = self.time_embedder(time)
            global_meta_mask = (node_type == 9)
            
            if global_meta_mask.any():
                h_new = h.clone()
                h_new[global_meta_mask] = h_new[global_meta_mask] + time_emb
                h = h_new

        # 0c. Edge Feature Processing
        if self.use_categorical_edge_types:
            if edge_type is None:
                # Fallback to puzzle type (0) if not provided
                edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
            
            # Embed categorical type
            edge_emb = self.edge_type_embedding(edge_type)
            
            # Concatenate with continuous features if they exist
            if edge_attr is not None:
                edge_attr = torch.cat([edge_emb, edge_attr], dim=-1)
            else:
                edge_attr = edge_emb
        elif edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim),
                                  device=x.device, dtype=torch.float)

        for conv, norm in zip(self.convs, self.norms, strict=True):
            h_in = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h)
            h = func.relu(h)
            h = func.dropout(h, p=self.dropout, training=self.training)

            if h_in.shape == h.shape:
                h = h + h_in

        edge_src, edge_dst = edge_index

        if self.edge_concat_component_meta:
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
            edge_features = torch.cat([h[edge_src], h[edge_dst]], dim=-1)

        if self.edge_concat_global_meta and self.use_meta_node:
            global_meta_mask = node_type == 9
            global_meta_emb = h[global_meta_mask]

            if batch is not None:
                edge_batch = batch[edge_src]
                global_emb_for_edges = global_meta_emb[edge_batch]
            else:
                global_emb_for_edges = global_meta_emb.expand(edge_src.size(0), -1)

            edge_features = torch.cat([edge_features, global_emb_for_edges], dim=-1)

        if self.use_edge_features_in_prediction:
            edge_features = torch.cat([edge_features, edge_attr], dim=-1)

        edge_logits = self.edge_mlp(edge_features)

        results = [edge_logits]

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
                    num_graphs = meta_embeddings.size(0)
                    pooled_puzzle = global_mean_pool(
                        puzzle_h, puzzle_batch, size=num_graphs
                    )
                else:
                    # Single graph case (batch is None)
                    if puzzle_h.size(0) > 0:
                        pooled_puzzle = puzzle_h.mean(dim=0, keepdim=True)
                    else:
                        pooled_puzzle = torch.zeros((1, h.size(-1)), device=h.device)

                verify_input = torch.cat([verify_input, pooled_puzzle], dim=-1)

            if self.verifier_use_row_col_meta_nodes:
                # Pool row/col meta nodes (node_type=10)
                meta_mask_extended = (node_type == 10)  # All row/col meta nodes
                meta_extended_h = h[meta_mask_extended]

                if batch is not None:
                    meta_extended_batch = batch[meta_mask_extended]
                    num_graphs = meta_embeddings.size(0)
                    pooled_meta_extended = global_mean_pool(
                        meta_extended_h, meta_extended_batch, size=num_graphs
                    )
                else:
                    if meta_extended_h.size(0) > 0:
                        pooled_meta_extended = meta_extended_h.mean(
                            dim=0, keepdim=True
                        )
                    else:
                        pooled_meta_extended = torch.zeros(
                            (1, h.size(-1)), device=h.device
                        )

                verify_input = torch.cat(
                    [verify_input, pooled_meta_extended, pooled_meta_extended], dim=-1
                )

            verify_logits = self.verify_mlp(verify_input)  # [num_graphs, 1]
            results.append(verify_logits)

        # 5. Diffusion auxiliary head (Consolidated Noise Head)
        if return_noise and self.use_noise_head:
            meta_mask = (node_type == 9)
            meta_embeddings = h[meta_mask]

            # --- Statistical Pooling (f.7) ---
            # Compute stats from edge_logits to help predict noise level
            # Convert to probs for entropy/confidence/margin
            probs = func.softmax(edge_logits, dim=-1)

            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1, keepdim=True)

            conf, _ = probs.max(dim=-1, keepdim=True)

            top2_probs, _ = probs.topk(2, dim=-1)
            margin = (top2_probs[:, 0:1] - top2_probs[:, 1:2])

            # Combine all raw signals for pooling
            # [Raw(3), Entropy(1), Conf(1), Margin(1)]
            raw_signals = torch.cat([edge_logits, entropy, conf, margin], dim=-1)

            # Pool Mean and Std across edges per graph
            if batch is not None:
                edge_batch = batch[edge_src]
                num_graphs = meta_embeddings.size(0)

                # Mean Pool
                mean_stats = global_mean_pool(raw_signals, edge_batch, size=num_graphs)

                # Std Pool (Var = E[X^2] - E[X]^2)
                mean_sq = global_mean_pool(raw_signals**2, edge_batch, size=num_graphs)
                std_stats = torch.sqrt(func.relu(mean_sq - mean_stats**2) + 1e-9)
            else:
                # Single graph case
                mean_stats = raw_signals.mean(dim=0, keepdim=True)
                std_stats = raw_signals.std(dim=0, keepdim=True)

            pooled_stats = torch.cat([mean_stats, std_stats], dim=-1)

            # Concatenate meta embedding with stats
            aux_input = torch.cat([meta_embeddings, pooled_stats], dim=-1)
            aux_logits = self.diffusion_aux_mlp(aux_input)
            results.append(aux_logits)

        if len(results) == 1:
            return results[0]
        return tuple(results)
