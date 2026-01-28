"""Factory for creating refactored Hashi Graph models."""

import torch
from .config import HashiModelConfig
from .features import NodeFeatureManager, EdgeFeatureManager
from .encoders import NodeEncoder, EdgeEncoder
from .backbone import GraphBackbone
from .heads import EdgeHead, ProphetHead
from .core import HashiGraphModel


class ModelFactory:
    """Factory class to create and configure model instances using component architecture."""

    @staticmethod
    def create_model(config: HashiModelConfig, device: torch.device) -> HashiGraphModel:
        """Create and return the integrated HashiGraphModel."""
        model_config = config.model
        
        # 1. Feature Managers
        node_fm = NodeFeatureManager(model_config)
        edge_fm = EdgeFeatureManager(model_config)
        
        # 2. Encoders
        node_encoder = NodeEncoder(model_config, node_fm)
        edge_encoder = EdgeEncoder(model_config, edge_fm)
        
        # 3. Calculate Dimensions
        # Node encoder outputs hidden_channels after its internal refiner
        node_hidden_dim = model_config.hidden_channels
        
        # Edge encoder output dimension
        edge_attr_dim = edge_encoder.output_dim
            
        # 4. Backbone
        # If noise is injected into MP, we must increase edge_dim for the backbone
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
        
        # 5. Heads
        edge_head = EdgeHead(
            model_config,
            node_hidden_dim=backbone.final_dim,
            edge_attr_dim=backbone_edge_dim
        )
        
        prophet_head = None
        if model_config.use_noise_head:
            prophet_head = ProphetHead(
                model_config,
                node_hidden_dim=backbone.final_dim
            )
            
        # 6. Assemble
        model = HashiGraphModel(
            config=config,
            node_encoder=node_encoder,
            edge_encoder=edge_encoder,
            backbone=backbone,
            edge_head=edge_head,
            prophet_head=prophet_head
        )
        
        return model.to(device)
