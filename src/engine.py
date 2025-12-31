"""
Core training engine for Hashi GNN.
Centralizes model creation, dataset loading, and training loop components.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List, Union

from .data import HashiDataset, HashiDatasetCache
from .models import (
    GCNEdgeClassifier, GATEdgeClassifier, 
    GINEEdgeClassifier, TransformerEdgeClassifier
)
from .train_utils import calculate_batch_perfect_puzzles, get_edge_batch_indices
from .utils import custom_collate_with_conflicts
from .losses import compute_combined_loss

class EpochMetrics:
    """Container for metrics returned from run_epoch."""
    def __init__(self):
        self.loss: float = 0.0
        self.accuracy: float = 0.0
        self.perfect_accuracy: float = 0.0
        self.ce_loss: float = 0.0
        self.degree_loss: float = 0.0
        self.crossing_loss: float = 0.0
        self.verify_loss: float = 0.0
        self.verify_balanced_acc: float = 0.0
        self.verify_recall_pos: float = 0.0
        self.verify_recall_neg: float = 0.0
    
    def to_tuple(self) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        """Return metrics as tuple for backward compatibility."""
        return (
            self.loss, self.ce_loss, self.degree_loss, self.crossing_loss, self.verify_loss,
            self.accuracy, self.perfect_accuracy, self.verify_balanced_acc, self.verify_recall_pos, self.verify_recall_neg
        )

class EarlyStopper:
    """Utility to signal when validation loss stops improving."""
    def __init__(self, patience: int = 1, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        """Return True once the monitored metric fails to improve."""
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_masking_rate(epoch: int, masking_config: Dict[str, Any], total_epochs: int) -> float:
    """Calculate progressive masking rate based on epoch."""
    if not masking_config.get('enabled', False):
        return 0.0
    
    warmup_epochs = masking_config.get('warmup_epochs', 0)
    cooldown_epochs = masking_config.get('cooldown_epochs', 0)
    start_rate = masking_config.get('start_rate', 0.0)
    end_rate = masking_config.get('end_rate', 1.0)
    schedule = masking_config.get('schedule', 'cosine')

    rampup_epochs = total_epochs - warmup_epochs - cooldown_epochs
    if rampup_epochs <= 0:
        return start_rate if epoch <= warmup_epochs else end_rate

    if epoch <= warmup_epochs:
        return start_rate
    if epoch > (warmup_epochs + rampup_epochs):
        return end_rate
    
    progress = (epoch - warmup_epochs) / rampup_epochs
    progress = min(progress, 1.0)
    
    if schedule == 'cosine':
        rate = start_rate + (end_rate - start_rate) * (1 - np.cos(np.pi * progress)) / 2
    elif schedule == 'linear':
        rate = start_rate + (end_rate - start_rate) * progress
    elif schedule == 'constant':
        rate = start_rate
    else:
        raise ValueError(f"Unknown masking schedule: {schedule}")
    
    return float(rate)

def apply_edge_label_masking(
    data: Data,
    masking_rate: float,
    device: torch.device,
    config: Dict[str, Any]
) -> Data:
    """Mask the bridge label and is_labeled features for a subset of edges."""
    if masking_rate <= 0.0 or data.edge_attr is None:
        return data

    edge_dim = data.edge_attr.size(1)
    bridge_label_idx = edge_dim - 2
    is_labeled_idx = edge_dim - 1

    if edge_dim < 2:
        return data

    model_config = config.get('model', {})
    use_capacity = model_config.get('use_capacity', True)
    use_structural_degree = model_config.get('use_structural_degree', True)
    use_structural_degree_nsew = model_config.get('use_structural_degree_nsew', False)
    use_unused_capacity = model_config.get('use_unused_capacity', True)

    unused_capacity_idx = 0
    if use_capacity: unused_capacity_idx += 1
    if use_structural_degree or use_structural_degree_nsew: unused_capacity_idx += 1

    data.edge_attr = data.edge_attr.clone()
    if use_unused_capacity:
        data.x = data.x.clone()

    original_edge_indices = torch.where(data.edge_mask)[0]
    num_to_mask = int(len(original_edge_indices) * masking_rate)

    if num_to_mask > 0:
        perm = torch.randperm(len(original_edge_indices), device=device)[:num_to_mask]
        mask_indices = original_edge_indices[perm]

        if use_unused_capacity:
            original_bridge_labels = data.edge_attr[mask_indices, bridge_label_idx].clone()

        data.edge_attr[mask_indices, bridge_label_idx] = 0.0
        data.edge_attr[mask_indices, is_labeled_idx] = 0.0

        if use_unused_capacity:
            src_nodes = data.edge_index[0, mask_indices]
            dst_nodes = data.edge_index[1, mask_indices]
            data.x[src_nodes, unused_capacity_idx] += original_bridge_labels
            data.x[dst_nodes, unused_capacity_idx] += original_bridge_labels

    return data

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

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    training: bool = True,
    optimizer: Optional[Optimizer] = None,
    masking_rate: float = 0.0,
    accumulation_steps: int = 1,
) -> EpochMetrics:
    """Unified function to run a training or evaluation epoch."""
    if training:
        if optimizer is None:
            raise ValueError("Optimizer required for training mode")
        model.train()
        desc = "Training"
    else:
        model.eval()
        desc = "Evaluating"
    
    total_loss = torch.tensor(0.0, device=device)
    total_ce_loss = torch.tensor(0.0, device=device)
    total_degree_loss = torch.tensor(0.0, device=device)
    total_crossing_loss = torch.tensor(0.0, device=device)
    total_verify_loss = torch.tensor(0.0, device=device)
    total_verify_acc = torch.tensor(0.0, device=device)
    total_verify_recall_pos = torch.tensor(0.0, device=device)
    total_verify_recall_neg = torch.tensor(0.0, device=device)
    correct_predictions = torch.tensor(0, device=device)
    total_edges = torch.tensor(0, device=device)
    perfect_puzzle_stats = []
    num_verify_batches = 0
    
    loss_weights = config['training'].get('loss_weights')
    use_verification = config['model'].get('use_verification_head', False)

    context = torch.no_grad() if not training else torch.enable_grad()
    
    with context:
        if training:
            optimizer.zero_grad()
        
        for batch_idx, data in enumerate(tqdm(loader, desc=desc)):
            data = data.to(device)
            
            if masking_rate > 0.0:
                data = apply_edge_label_masking(data, masking_rate, device, config)
            
            edge_attr = getattr(data, 'edge_attr', None)
            edge_batch = get_edge_batch_indices(data)
            
            model_has_verify = hasattr(model, 'use_verification_head') and model.use_verification_head
            should_verify = use_verification and model_has_verify
            
            if should_verify:
                logits, verify_logits = model(
                    data.x, data.edge_index, edge_attr=edge_attr,
                    batch=getattr(data, 'batch', None), return_verification=True
                )
            else:
                logits = model(
                    data.x, data.edge_index, edge_attr=edge_attr,
                    batch=getattr(data, 'batch', None)
                )
                verify_logits = None
            
            node_capacities = data.x[:, 0].long()
            edge_conflicts = getattr(data, 'edge_conflicts', None)
            
            losses = compute_combined_loss(
                logits, data.y, data.edge_index, node_capacities,
                edge_conflicts, data.edge_mask, loss_weights,
                verify_logits=verify_logits,
                edge_batch=edge_batch
            )
            loss = losses['total']
            
            total_ce_loss += losses['ce'] * data.num_graphs
            total_degree_loss += losses['degree'] * data.num_graphs
            total_crossing_loss += losses['crossing'] * data.num_graphs
            total_verify_loss += losses['verify'] * data.num_graphs
            total_verify_acc += losses['verify_acc']
            total_verify_recall_pos += losses['verify_recall_pos']
            total_verify_recall_neg += losses['verify_recall_neg']
            if losses['verify'] > 0:
                num_verify_batches += 1
            
            if training:
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()
            
            logits_original = logits[data.edge_mask]
            total_loss += loss * data.num_graphs
            pred = logits_original.argmax(dim=-1)
            correct_predictions += (pred == data.y).sum()
            total_edges += data.edge_mask.sum()
            
            edge_batch_original = edge_batch[data.edge_mask]
            _, num_perfect, num_total = calculate_batch_perfect_puzzles(
                logits_original, data.y,
                torch.ones(logits_original.size(0), dtype=torch.bool, device=device),
                edge_batch_original
            )
            perfect_puzzle_stats.append((num_perfect, num_total))
    
    num_samples = len(loader.dataset)
    metrics = EpochMetrics()
    metrics.loss = (total_loss / num_samples).item()
    metrics.ce_loss = (total_ce_loss / num_samples).item()
    metrics.degree_loss = (total_degree_loss / num_samples).item()
    metrics.crossing_loss = (total_crossing_loss / num_samples).item()
    metrics.verify_loss = (total_verify_loss / num_samples).item()
    metrics.verify_balanced_acc = (total_verify_acc / num_verify_batches).item() if num_verify_batches > 0 else 0.0
    metrics.verify_recall_pos = (total_verify_recall_pos / num_verify_batches).item() if num_verify_batches > 0 else 0.0
    metrics.verify_recall_neg = (total_verify_recall_neg / num_verify_batches).item() if num_verify_batches > 0 else 0.0
    metrics.accuracy = (correct_predictions / total_edges).item()
    
    total_perfect = sum(p for p, _ in perfect_puzzle_stats)
    total_puzzles = sum(t for _, t in perfect_puzzle_stats)
    metrics.perfect_accuracy = total_perfect / total_puzzles if total_puzzles > 0 else 0.0
    
    return metrics

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
