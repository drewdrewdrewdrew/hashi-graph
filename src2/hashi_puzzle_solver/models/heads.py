"""Prediction heads for Hashi Puzzle Solver."""

import torch
from torch.nn import Dropout, Linear, ReLU, Sequential, LayerNorm
import torch.nn.functional as func
from torch_geometric.nn import global_mean_pool

from .config import ModelConfig
from .common import build_mlp


class EdgeHead(torch.nn.Module):
    """
    Prediction head for edge bridge counts (0, 1, or 2).
    Handles feature concatenation for edge-wise classification.
    """

    def __init__(
        self,
        config: ModelConfig,
        node_hidden_dim: int,
        edge_attr_dim: int,
    ):
        super().__init__()
        self.config = config
        
        # Calculate input dimension for edge MLP
        # Base: Source node + Destination node
        input_dim = 2 * node_hidden_dim
        
        if config.edge_concat_global_meta:
            input_dim += node_hidden_dim
            
        if config.use_component_meta:
            # We concatenate 2 * node_hidden_dim for source and target component metas
            input_dim += 2 * node_hidden_dim
            
        if config.use_edge_features_in_prediction:
            input_dim += edge_attr_dim

        if config.use_noise_in_prediction:
            input_dim += config.noise_embedding_dim

        # Dynamic MLP construction
        hidden_dim = int(round(node_hidden_dim * config.edge_mlp_width_mult))
        self.mlp = build_mlp(
            input_dim=input_dim,
            output_dim=3,
            hidden_dim=hidden_dim,
            num_layers=config.edge_mlp_depth_mult,
            dropout=config.dropout,
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        node_type: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        noise_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict bridge counts for each edge."""
        edge_src, edge_dst = edge_index
        src_h = h[edge_src]
        dst_h = h[edge_dst]
        
        features = [src_h, dst_h]
        
        # 1. Component Meta Concatenation
        if self.config.use_component_meta and node_type is not None:
            # Identify edges from islands (type <= 8) to component metas (type 11)
            # edge_index contains all edges, including those connecting islands to their metas
            comp_edges_mask = (node_type[edge_index[0]] <= 8) & (node_type[edge_index[1]] == 11)
            
            # Map each island to its component meta node index
            island_to_comp_meta = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            island_to_comp_meta[edge_index[0, comp_edges_mask]] = edge_index[1, comp_edges_mask]
            
            # Fetch meta embeddings for source and destination islands of the edges we're predicting for
            src_meta_h = h[island_to_comp_meta[edge_src]]
            dst_meta_h = h[island_to_comp_meta[edge_dst]]
            
            features.insert(1, src_meta_h)
            features.insert(3, dst_meta_h)

        # 2. Global Meta Concatenation
        if self.config.edge_concat_global_meta and node_type is not None:
            global_meta_mask = (node_type == 9)
            global_meta_emb = h[global_meta_mask]
            
            if global_meta_emb.size(0) > 0:
                if batch is not None:
                    edge_batch = batch[edge_src]
                    global_emb_for_edges = global_meta_emb[edge_batch]
                else:
                    global_emb_for_edges = global_meta_emb.expand(edge_src.size(0), -1)
                features.append(global_emb_for_edges)

        # 3. Edge Attribute Concatenation
        if self.config.use_edge_features_in_prediction and edge_attr is not None:
            features.append(edge_attr)

        # 4. Noise Embedding Concatenation
        if self.config.use_noise_in_prediction:
            if noise_emb is not None:
                if batch is not None:
                    edge_batch = batch[edge_src]
                    noise_for_edges = noise_emb[edge_batch]
                else:
                    noise_for_edges = noise_emb.expand(edge_src.size(0), -1)
            else:
                # Provide zero embeddings if noise is enabled but not provided
                noise_for_edges = torch.zeros(
                    (edge_src.size(0), self.config.noise_embedding_dim),
                    device=h.device,
                    dtype=h.dtype
                )
            features.append(noise_for_edges)
            
        edge_features = torch.cat(features, dim=-1)
        return self.mlp(edge_features)


class ProphetHead(torch.nn.Module):
    """
    Consolidated noise prediction head (sigma, alpha) with statistical pooling.
    """

    def __init__(
        self,
        config: ModelConfig,
        node_hidden_dim: int,
    ):
        super().__init__()
        self.config = config
        self.stats_dim = 12  # [RawLogits(3), Entropy(1), Confidence(1), Margin(1)] * 2 (Mean, Std)
        
        input_dim = node_hidden_dim + self.stats_dim
        hidden_dim = int(round(node_hidden_dim * config.noise_mlp_width_mult))
        
        self.mlp = build_mlp(
            input_dim=input_dim,
            output_dim=2,
            hidden_dim=hidden_dim,
            num_layers=config.noise_mlp_depth_mult,
            dropout=config.dropout,
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_logits: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise parameters (sigma, alpha)."""
        edge_src, _ = edge_index
        
        # 1. Global Meta Embedding
        meta_mask = (node_type == 9)
        meta_embeddings = h[meta_mask]
        
        # 2. Statistical Pooling
        probs = func.softmax(edge_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1, keepdim=True)
        conf, _ = probs.max(dim=-1, keepdim=True)
        top2_probs, _ = probs.topk(2, dim=-1)
        margin = (top2_probs[:, 0:1] - top2_probs[:, 1:2])
        
        raw_signals = torch.cat([edge_logits, entropy, conf, margin], dim=-1)
        
        if batch is not None:
            edge_batch = batch[edge_src]
            num_graphs = meta_embeddings.size(0)
            mean_stats = global_mean_pool(raw_signals, edge_batch, size=num_graphs)
            mean_sq = global_mean_pool(raw_signals**2, edge_batch, size=num_graphs)
            std_stats = torch.sqrt(func.relu(mean_sq - mean_stats**2) + 1e-9)
        else:
            mean_stats = raw_signals.mean(dim=0, keepdim=True)
            std_stats = raw_signals.std(dim=0, keepdim=True)
            
        pooled_stats = torch.cat([mean_stats, std_stats], dim=-1)
        
        # 3. Final Prediction
        aux_input = torch.cat([meta_embeddings, pooled_stats], dim=-1)
        return self.mlp(aux_input)
