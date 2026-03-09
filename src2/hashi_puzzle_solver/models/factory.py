"""Factory for creating refactored Hashi Graph models."""

import torch
from .config import HashiModelConfig
from .features import NodeFeatureManager, EdgeFeatureManager
from .encoders import NodeEncoder, EdgeEncoder
from .backbone import GraphBackbone
from .heads import EdgeHead, ProphetHead
from .core import HashiGraphModel
from .iterative_backbone import IterativeBackbone
from .reverse_backbone import ReverseBackbone


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
        
        # 5. Optional components (Phase 5)
        iterative_bb: IterativeBackbone | None = None
        if model_config.reasoning.enabled:
            iterative_bb = IterativeBackbone(
                hidden_channels=model_config.hidden_channels,
                steps=model_config.reasoning.steps,
                heads=model_config.heads,
                dropout=model_config.dropout,
                edge_dim=backbone_edge_dim,
            )

        reverse_bb: ReverseBackbone | None = None
        if model_config.reverse_gnn.enabled:
            reverse_bb = ReverseBackbone(
                forward_backbone=backbone,
                hidden_channels=model_config.hidden_channels,
                separate_weights=model_config.reverse_gnn.separate_weights,
                project_embeddings=model_config.reverse_gnn.project_embeddings,
            )

        # Compute node_hidden_dim for EdgeHead based on active flag combination
        edge_head_node_dim = backbone.final_dim
        if model_config.reverse_gnn.enabled:
            if model_config.reverse_gnn.project_embeddings:
                edge_head_node_dim = model_config.hidden_channels
            else:
                edge_head_node_dim = 2 * backbone.final_dim
        # reasoning.enabled alone does not change node embedding dim

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
            reverse_backbone=reverse_bb,
        )
        
        return model.to(device)
