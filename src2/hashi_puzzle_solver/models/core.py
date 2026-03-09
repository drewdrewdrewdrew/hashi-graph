"""Core model shell for Hashi Puzzle Solver."""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential

from .config import HashiModelConfig
from .backbone import GraphBackbone
from .encoders import NodeEncoder, EdgeEncoder
from .heads import EdgeHead, ProphetHead
from .iterative_backbone import IterativeBackbone
from .reverse_backbone import ReverseBackbone


class HashiGraphModel(torch.nn.Module):
    """
    Unified GNN model for Hashi puzzle solving.
    Assembles encoders, backbone, and heads into a single shell.
    """

    def __init__(
        self,
        config: HashiModelConfig,
        node_encoder: NodeEncoder,
        edge_encoder: EdgeEncoder,
        backbone: GraphBackbone,
        edge_head: EdgeHead,
        prophet_head: ProphetHead | None = None,
        verify_head: torch.nn.Module | None = None,
        iterative_backbone: IterativeBackbone | None = None,
        reverse_backbone: ReverseBackbone | None = None,
    ):
        super().__init__()
        self.config = config
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.backbone = backbone
        self.edge_head = edge_head
        self.prophet_head = prophet_head
        self.verify_head = verify_head
        self.iterative_backbone = iterative_backbone   # registered as submodule (nn.Module or None)
        self.reverse_backbone = reverse_backbone

        # Validate interleaved constraint at construction time
        if iterative_backbone is not None and reverse_backbone is not None:
            if not reverse_backbone.project_embeddings:
                raise ValueError(
                    "When both reasoning and reverse_gnn are enabled, "
                    "project_embeddings must be True (required for residual dimension match). "
                    "Set model.reverse_gnn.project_embeddings: true in config."
                )

        # Diffusion specific components
        # We create noise_projection if any noise injection is enabled
        use_noise = (
            config.model.use_noise_head or 
            config.model.use_noise_in_prediction or 
            config.model.use_noise_in_message_passing or 
            config.model.use_noise_in_global_meta
        )
        if use_noise:
            self.noise_projection = Sequential(
                Linear(2, config.model.noise_embedding_dim),
                ReLU(),
                torch.nn.LayerNorm(config.model.noise_embedding_dim),
            )
            
            if config.model.use_noise_in_global_meta:
                # Project from noise_emb_dim back to backbone.final_dim for additive injection
                self.noise_to_meta = Linear(config.model.noise_embedding_dim, backbone.final_dim)
        
        # Time conditioning
        # We'll use a simple time embedder if needed, or it can be a separate component.
        # For parity with original TransformerEdgeClassifier:
        # self.time_embedder = torch.nn.Sequential(...)
        # I'll add it here for now if the config says so.
        # But wait, ModelConfig doesn't have use_time_conditioning yet in my dataclass.
        # Let's check the original code again.
        # Ah, it was in TransformerEdgeClassifier.__init__ but not in my ModelConfig.
        # I'll add a simplified version if I see it in kwargs or if I want to support it.

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
        """Full forward pass."""
        # 1. Encode Nodes
        h = self.node_encoder(x)
        
        # 2. Encode Edges
        h_edge = self.edge_encoder(edge_attr, edge_type)
        
        # 3. Project Noise
        noise_emb = None
        if input_noise is not None and hasattr(self, "noise_projection"):
            noise_emb = self.noise_projection(input_noise)

        # 4. Inject Noise / Time into Global Meta Node
        if (noise_emb is not None or time is not None) and node_type is not None:
            global_meta_mask = (node_type == 9)
            if global_meta_mask.any():
                h_new = h.clone()
                if noise_emb is not None and self.config.model.use_noise_in_global_meta:
                    h_new[global_meta_mask] = h_new[global_meta_mask] + self.noise_to_meta(noise_emb)
                # (Add time conditioning here if supported)
                h = h_new

        # 5. Inject Noise into Edge Attributes for Message Passing
        if self.config.model.use_noise_in_message_passing:
            edge_src, _ = edge_index
            if noise_emb is not None:
                if batch is not None:
                    edge_batch = batch[edge_src]
                    noise_for_edges = noise_emb[edge_batch]
                else:
                    noise_for_edges = noise_emb.expand(edge_src.size(0), -1)
            else:
                # Provide zero embeddings if noise is enabled but not provided
                noise_for_edges = torch.zeros(
                    (edge_src.size(0), self.config.model.noise_embedding_dim),
                    device=h.device,
                    dtype=h_edge.dtype
                )
            h_edge = torch.cat([h_edge, noise_for_edges], dim=-1)

        # 6. Message Passing (Backbone) + optional composition
        h = self.backbone(h, edge_index, edge_attr=h_edge)

        if self.iterative_backbone is not None and self.reverse_backbone is not None:
            # Interleaved rev-reasoning: forward conv + reverse pass + project before each residual
            for _ in range(self.iterative_backbone.steps):
                h_in = h
                h_fwd = self.iterative_backbone.conv(h, edge_index, edge_attr=h_edge)
                h_fwd = self.iterative_backbone.norm(h_fwd)
                h_fwd = F.relu(h_fwd)
                h_fwd = F.dropout(h_fwd, p=self.iterative_backbone.dropout, training=self.training)
                h_rev = self.reverse_backbone(h, edge_index, edge_attr=h_edge)
                h_cat = torch.cat([h_fwd, h_rev], dim=-1)
                h = self.reverse_backbone.projection(h_cat)  # project_embeddings guaranteed True
                h = h + h_in
        elif self.iterative_backbone is not None:
            h = self.iterative_backbone(h, edge_index, edge_attr=h_edge)
        elif self.reverse_backbone is not None:
            h_rev = self.reverse_backbone(h, edge_index, edge_attr=h_edge)
            h = torch.cat([h, h_rev], dim=-1)
            if self.reverse_backbone.project_embeddings:
                h = self.reverse_backbone.projection(h)
        # else: h passes unchanged — baseline path
        
        # 7. Prediction Heads
        edge_logits = self.edge_head(
            h, 
            edge_index, 
            edge_attr=h_edge, 
            node_type=node_type, 
            batch=batch,
            noise_emb=noise_emb,
        )
        
        results = [edge_logits]
        
        if return_verification and self.verify_head is not None:
            # We'll need to define how verify_head works. 
            # For now, we'll assume it's a module that takes h and node_type.
            # (Simplifying for parity with original logic)
            pass
            
        if return_noise and self.prophet_head is not None:
            noise_logits = self.prophet_head(
                h, 
                edge_logits, 
                edge_index, 
                node_type=node_type, 
                batch=batch
            )
            results.append(noise_logits)
            
        if len(results) == 1:
            return results[0]
        return tuple(results)
