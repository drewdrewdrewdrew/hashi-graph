"""Factory for creating refactored Hashi Graph models."""

import torch
from .config import HashiModelConfig
from .features import NodeFeatureManager, EdgeFeatureManager
from .encoders import NodeEncoder, EdgeEncoder
from .backbone import GraphBackbone
from .heads import EdgeHead, ProphetHead
from .core import HashiGraphModel
from .iterative_backbone import IterativeBackbone


class ModelFactory:
    """Factory class to create and configure model instances using component architecture."""

    @staticmethod
    def create_model(config: HashiModelConfig, device: torch.device) -> HashiGraphModel:
        """Create and return the integrated HashiGraphModel."""
        model_config = config.model

        if model_config.reverse_gnn.enabled and not model_config.reasoning.enabled:
            raise ValueError(
                "reverse_gnn requires reasoning to be enabled "
                "(reverse reuses the iterative backbone's shared conv weights)."
            )

        # 1. Feature Managers
        node_fm = NodeFeatureManager(model_config)
        edge_fm = EdgeFeatureManager(model_config)
        
        # 2. Encoders
        node_encoder = NodeEncoder(model_config, node_fm)
        edge_encoder = EdgeEncoder(model_config, edge_fm)
        
        # 3. Calculate Dimensions
        node_hidden_dim = model_config.hidden_channels
        edge_attr_dim = edge_encoder.output_dim
            
        # 4. Backbone
        backbone_edge_dim = edge_attr_dim
        if model_config.use_noise_in_message_passing:
            backbone_edge_dim += model_config.noise_embedding_dim

        backbone = GraphBackbone(
            node_input_dim=node_hidden_dim,
            hidden_channels=model_config.hidden_channels,
            num_layers=model_config.num_layers,
            heads=model_config.heads,
            dropout=model_config.dropout,
            edge_dim=backbone_edge_dim,
            gnn_type=model_config.type
        )
        
        # 5. Optional iterative backbone (reasoning + optional reverse)
        iterative_bb: IterativeBackbone | None = None
        if model_config.reasoning.enabled:
            iterative_bb = IterativeBackbone(
                hidden_channels=model_config.hidden_channels,
                steps=model_config.reasoning.steps,
                heads=model_config.heads,
                dropout=model_config.dropout,
                edge_dim=backbone_edge_dim,
                reverse_enabled=model_config.reverse_gnn.enabled,
            )

        # Projection always maps back to hidden_channels
        edge_head_node_dim = backbone.final_dim

        # 6. Heads
        edge_head = EdgeHead(
            model_config,
            node_hidden_dim=edge_head_node_dim,
            edge_attr_dim=backbone_edge_dim
        )

        prophet_head = None
        if model_config.use_noise_head:
            prophet_head = ProphetHead(
                model_config,
                node_hidden_dim=edge_head_node_dim
            )

        # 7. Assemble
        model = HashiGraphModel(
            config=config,
            node_encoder=node_encoder,
            edge_encoder=edge_encoder,
            backbone=backbone,
            edge_head=edge_head,
            prophet_head=prophet_head,
            iterative_backbone=iterative_bb,
        )
        
        return model.to(device)
