"""Core model shell for Hashi Puzzle Solver."""

import torch
from torch.nn import Linear, ReLU, Sequential

from .config import HashiModelConfig
from .backbone import GraphBackbone
from .encoders import NodeEncoder, EdgeEncoder
from .heads import EdgeHead, ProphetHead


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
    ):
        super().__init__()
        self.config = config
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.backbone = backbone
        self.edge_head = edge_head
        self.prophet_head = prophet_head
        self.verify_head = verify_head

        # Diffusion specific components
        if config.model.use_noise_head:
            self.noise_embedder = Linear(2, backbone.final_dim)
        
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
        
        # 3. Inject Noise / Time into Global Meta Node
        if (input_noise is not None or time is not None) and node_type is not None:
            global_meta_mask = (node_type == 9)
            if global_meta_mask.any():
                h_new = h.clone()
                if input_noise is not None and hasattr(self, "noise_embedder"):
                    h_new[global_meta_mask] = h_new[global_meta_mask] + self.noise_embedder(input_noise)
                # (Add time conditioning here if supported)
                h = h_new

        # 4. Message Passing (Backbone)
        h = self.backbone(h, edge_index, edge_attr=h_edge)
        
        # 5. Prediction Heads
        edge_logits = self.edge_head(
            h, 
            edge_index, 
            edge_attr=h_edge, 
            node_type=node_type, 
            batch=batch
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
