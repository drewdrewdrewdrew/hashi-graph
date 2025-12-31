"""
Core training engine for Hashi GNN.
Centralizes model creation, dataset loading, and training loop components.
"""
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from .data import HashiDataset, HashiDatasetCache
from .models import (
    GCNEdgeClassifier, GATEdgeClassifier, 
    GINEEdgeClassifier, TransformerEdgeClassifier
)
from .train import custom_collate_with_conflicts

def compute_edge_dim(model_config: Dict[str, Any]) -> int:
    """Calculate edge dimension based on enabled features."""
    edge_dim = 3  # base: [inv_dx, inv_dy, is_meta]
    if model_config.get('use_conflict_edges', False):
        edge_dim += 1
    if model_config.get('use_meta_mesh', False):
        edge_dim += 1
    if model_config.get('use_meta_row_col_edges', False):
        edge_dim += 1
    if model_config.get('use_edge_labels_as_features', False):
        edge_dim += 2
    return edge_dim

def create_model(model_config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Centralized model factory."""
    model_type = model_config.get('type', 'gcn').lower()
    edge_dim = compute_edge_dim(model_config)
    
    # Common kwargs for all models
    common_kwargs = {
        'node_embedding_dim': model_config['node_embedding_dim'],
        'hidden_channels': model_config['hidden_channels'],
        'num_layers': model_config['num_layers'],
        'dropout': model_config.get('dropout', 0.25),
        'use_capacity': model_config.get('use_capacity', True),
        'use_structural_degree': model_config.get('use_structural_degree', True),
        'use_structural_degree_nsew': model_config.get('use_structural_degree_nsew', False),
        'use_unused_capacity': model_config.get('use_unused_capacity', True),
        'use_conflict_status': model_config.get('use_conflict_status', True),
        'use_meta_node': model_config.get('use_global_meta_node', True),
        'use_closeness': model_config.get('use_closeness_centrality', False),
    }

    if model_type == 'gcn':
        model = GCNEdgeClassifier(**common_kwargs)
    elif model_type == 'gat':
        model = GATEdgeClassifier(
            **common_kwargs,
            heads=model_config.get('heads', 8),
            use_row_col_meta=model_config.get('use_row_col_meta', False),
            edge_dim=edge_dim
        )
    elif model_type == 'gine':
        model = GINEEdgeClassifier(
            **common_kwargs,
            use_row_col_meta=model_config.get('use_row_col_meta', False),
            edge_dim=edge_dim
        )
    elif model_type == 'transformer':
        model = TransformerEdgeClassifier(
            **common_kwargs,
            heads=model_config.get('heads', 4),
            use_row_col_meta=model_config.get('use_row_col_meta', False),
            edge_dim=edge_dim,
            use_verification_head=model_config.get('use_verification_head', False),
            verifier_use_puzzle_nodes=model_config.get('verifier_use_puzzle_nodes', False),
            verifier_use_row_col_meta_nodes=model_config.get('verifier_use_row_col_meta_nodes', False),
            edge_concat_global_meta=model_config.get('edge_concat_global_meta', False)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)

def create_dataloader(
    config: Dict[str, Any], 
    split: str, 
    transform: Optional[Any] = None,
    use_cache: bool = False
) -> DataLoader:
    """Centralized dataloader factory."""
    data_config = config['data']
    model_config = config['model']
    
    if use_cache:
        dataset = HashiDatasetCache.get_or_create(config, split, transform=transform)
    else:
        dataset = HashiDataset(
            root=Path(data_config['root_dir']),
            split=split,
            size=data_config.get('size'),
            difficulty=data_config.get('difficulty'),
            limit=data_config.get('limit'),
            use_degree=model_config.get('use_degree', False),
            use_meta_node=model_config.get('use_global_meta_node', True),
            use_row_col_meta=model_config.get('use_row_col_meta', False),
            use_meta_mesh=model_config.get('use_meta_mesh', False),
            use_meta_row_col_edges=model_config.get('use_meta_row_col_edges', False),
            use_distance=model_config.get('use_distance', False),
            use_edge_labels_as_features=model_config.get('use_edge_labels_as_features', False),
            use_closeness_centrality=model_config.get('use_closeness_centrality', False),
            use_conflict_edges=model_config.get('use_conflict_edges', False),
            use_capacity=model_config.get('use_capacity', True),
            use_structural_degree=model_config.get('use_structural_degree', True),
            use_structural_degree_nsew=model_config.get('use_structural_degree_nsew', False),
            use_unused_capacity=model_config.get('use_unused_capacity', True),
            use_conflict_status=model_config.get('use_conflict_status', True),
            transform=transform
        )

    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['training'].get('num_workers', 0),
        collate_fn=custom_collate_with_conflicts,
        persistent_workers=config['training'].get('use_persistent_workers', False)
    )

