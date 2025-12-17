"""
Main script for training and evaluating the GNN model for edge classification.
"""
import argparse
from contextlib import nullcontext
from pathlib import Path
import yaml
import mlflow
import datetime
import numpy as np
import torch
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

from .data import HashiDataset
from .models import GCNEdgeClassifier, GATEdgeClassifier, GINEEdgeClassifier, TransformerEdgeClassifier
from .train_utils import calculate_batch_perfect_puzzles, get_edge_batch_indices
from .losses import compute_combined_loss


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def flatten_config(config: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary config."""
    items: List[Tuple[str, Any]] = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_device(device_config: str) -> torch.device:
    """Determine the compute device based on config and availability."""
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_config)


def _normalize_conflict_index(value: Any) -> int:
    """
    Normalize a conflict entry to a concrete edge index.

    Args:
        value: Potential representations from tensors, nested tuples/lists, or scalar values.

    Returns:
        The integer index for the edge conflict entry.

    Raises:
        ValueError: If the value cannot be coerced into an integer index.
    """
    if isinstance(value, torch.Tensor):
        value = value.item()

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Edge conflict entries must be non-empty.")
        return _normalize_conflict_index(value[0])

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Edge conflict entry {value!r} cannot be converted to an integer index."
        ) from exc


def _normalize_conflict_pair(conflict: Any) -> Tuple[int, int]:
    """
    Normalize a potential edge conflict entry into a pair of integer indices.

    Args:
        conflict: Value that should describe two edges, e.g., a list/tuple of ints or tensors.

    Returns:
        Tuple of (edge_index_1, edge_index_2).

    Raises:
        ValueError: If the entry cannot be reduced to exactly two integer indices.
    """
    if isinstance(conflict, torch.Tensor):
        conflict = conflict.tolist()

    if not isinstance(conflict, (list, tuple)) or len(conflict) != 2:
        raise ValueError("Each edge_conflict must contain exactly two entries.")

    first_idx = _normalize_conflict_index(conflict[0])
    second_idx = _normalize_conflict_index(conflict[1])

    return first_idx, second_idx


def get_masking_rate(epoch: int, masking_config: Dict[str, Any], total_epochs: int) -> float:
    """
    Calculate progressive masking rate based on epoch, including warmup and cooldown periods.
    
    Args:
        epoch: Current epoch (1-indexed)
        masking_config: Dictionary containing masking parameters.
        total_epochs: Total number of epochs for the entire training run.
        
    Returns:
        float: The masking rate for the current epoch (0.0 to 1.0).
    """
    if not masking_config.get('enabled', False):
        return 0.0
    
    warmup_epochs = masking_config.get('warmup_epochs', 0)
    cooldown_epochs = masking_config.get('cooldown_epochs', 0)
    
    # Calculate effective epochs for ramping up masking
    rampup_epochs = total_epochs - warmup_epochs - cooldown_epochs
    if rampup_epochs <= 0:
        # If no rampup period, masking is either 0 or 1 depending on config
        if epoch < warmup_epochs:
            return 0.0
        else:
            return masking_config.get('end_rate', 1.0) # Assume full masking if no rampup

    # Handle warmup phase
    if epoch <= warmup_epochs:
        return 0.0
    
    # Handle cooldown phase (after rampup, before end of training)
    if epoch > (warmup_epochs + rampup_epochs):
        return masking_config.get('end_rate', 1.0) # Maintain full masking during cooldown
    
    # Calculate progress during rampup phase
    start_rate = masking_config.get('start_rate', 0.0)
    end_rate = masking_config.get('end_rate', 1.0)
    schedule = masking_config.get('schedule', 'cosine')
    
    # Normalize epoch to the rampup period
    progress = (epoch - warmup_epochs) / rampup_epochs
    progress = min(progress, 1.0)  # Ensure it doesn't exceed 1.0
    
    if schedule == 'cosine':
        rate = start_rate + (end_rate - start_rate) * (1 - np.cos(np.pi * progress)) / 2
    elif schedule == 'linear':
        rate = start_rate + (end_rate - start_rate) * progress
    elif schedule == 'constant': # Added constant schedule as per config comment
        rate = start_rate # Should be start_rate for the entire rampup if constant
    else:
        raise ValueError(f"Unknown masking schedule: {schedule}. Choose 'cosine', 'linear', or 'constant'.")
    
    return float(rate)


def apply_edge_label_masking(data: Data, masking_rate: float, device: torch.device) -> Data:
    """
    Mask the bridge label and is_labeled features for a subset of edges.
    
    Args:
        data: Batched graph data.
        masking_rate: Fraction of original edges to mask.
        device: Device to run masking on (used for rng placement).

    Returns:
        Data with masked features (in-place).
    """
    if masking_rate <= 0.0 or data.edge_attr is None:
        return data
    
    edge_dim = data.edge_attr.size(1)
    
    # Bridge label is always second-to-last, is_labeled is always last
    bridge_label_idx = edge_dim - 2
    is_labeled_idx = edge_dim - 1
    
    # Minimum edge_attr must have at least these two features
    if edge_dim < 2:
        return data
    
    # Clone to avoid modifying the original tensor
    data.edge_attr = data.edge_attr.clone()
    
    # Only mask original puzzle edges (not meta edges)
    original_edge_indices = torch.where(data.edge_mask)[0]
    num_to_mask = int(len(original_edge_indices) * masking_rate)
    
    if num_to_mask > 0:
        perm = torch.randperm(len(original_edge_indices), device=device)[:num_to_mask]
        mask_indices = original_edge_indices[perm]
        
        # Zero out bridge_label and is_labeled flag using dynamic indices
        data.edge_attr[mask_indices, bridge_label_idx] = 0.0
        data.edge_attr[mask_indices, is_labeled_idx] = 0.0
    
    return data


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


def custom_collate_with_conflicts(data_list: List[Data]) -> Batch:
    """
    Custom collate function that properly handles edge_conflicts during batching.
    
    PyTorch Geometric's default batching doesn't handle custom list attributes,
    and edge indices need to be offset when combining multiple graphs.
    
    Args:
        data_list: List of Data objects to batch
    
    Returns:
        Batched Data object with properly adjusted edge_conflicts
    """
    # Extract edge_conflicts before batching to prevent PyG from mangling them
    conflicts_per_graph = [getattr(data, 'edge_conflicts', None) for data in data_list]
    
    # Use PyG's standard batching, excluding edge_conflicts to handle manually
    batch = Batch.from_data_list(data_list, exclude_keys=['edge_conflicts'])
    
    # Manually handle edge_conflicts with proper index offsetting
    all_conflicts = []
    edge_offset = 0
    
    for graph_idx, data in enumerate(data_list):
        conflicts = conflicts_per_graph[graph_idx]
        if conflicts:
            num_edges = data.edge_index.size(1)
            for conflict in conflicts:
                if not conflict:
                    raise ValueError(
                        f"Empty edge_conflict entry found for graph index {graph_idx}."
                    )

                try:
                    e1, e2 = _normalize_conflict_pair(conflict)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid edge_conflict entry at graph index {graph_idx}: {conflict}"
                    ) from exc

                if not (0 <= e1 < num_edges and 0 <= e2 < num_edges):
                    raise ValueError(
                        f"Edge conflict indices {e1}, {e2} out of range for graph {graph_idx} "
                        f"with {num_edges} edges."
                    )

                all_conflicts.append((e1 + edge_offset, e2 + edge_offset))
        
        # Update offset for next graph (each graph contributes this many edges)
        edge_offset += data.edge_index.size(1)
    
    # Store the properly batched conflicts
    batch.edge_conflicts = all_conflicts if all_conflicts else None
    
    return batch


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
        self.verify_accuracy: float = 0.0
    
    def to_tuple(self) -> Tuple[float, float, float, float, float, float, float, float]:
        """Return metrics as tuple for backward compatibility."""
        return (
            self.loss, self.accuracy, self.perfect_accuracy,
            self.ce_loss, self.degree_loss, self.crossing_loss,
            self.verify_loss, self.verify_accuracy
        )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    training: bool = True,
    optimizer: Optional[Optimizer] = None,
    criterion: Optional[_Loss] = None,
    masking_rate: float = 0.0,
    loss_weights: Optional[Dict[str, float]] = None,
    use_verification: bool = False,
) -> EpochMetrics:
    """
    Unified function to run a training or evaluation epoch.
    
    Args:
        model: The GNN model
        loader: DataLoader for the dataset
        device: Torch device
        training: If True, run in training mode (backward pass + optimizer step)
        optimizer: Optimizer (required if training=True)
        criterion: Loss criterion for fallback CE loss
        masking_rate: Fraction of original edges to mask
        loss_weights: Dict with 'ce', 'degree', 'crossing', 'verify' weights
        use_verification: Whether to compute verification loss (requires model with verification head)
    
    Returns:
        EpochMetrics containing all loss and accuracy metrics
    """
    if training:
        if optimizer is None:
            raise ValueError("Optimizer required for training mode")
        model.train()
        desc = "Training"
    else:
        model.eval()
        desc = "Evaluating"
    
    # Initialize accumulators
    total_loss = 0.0
    total_ce_loss = 0.0
    total_degree_loss = 0.0
    total_crossing_loss = 0.0
    total_verify_loss = 0.0
    total_verify_acc = 0.0
    correct_predictions = 0
    total_edges = 0
    perfect_puzzle_stats = []
    num_verify_batches = 0
    
    # Context manager for eval mode
    context = torch.no_grad() if not training else nullcontext()
    
    with context:
        for batch_idx, data in enumerate(tqdm(loader, desc=desc)):
            data = data.to(device)
            
            # Apply edge label masking if enabled
            if masking_rate > 0.0:
                data = apply_edge_label_masking(data, masking_rate, device)
            
            if training:
                optimizer.zero_grad()
            
            # Forward pass
            edge_attr = getattr(data, 'edge_attr', None)
            edge_batch = get_edge_batch_indices(data)
            
            # Check if model supports verification
            model_has_verify = hasattr(model, 'use_verification_head') and model.use_verification_head
            should_verify = use_verification and model_has_verify
            
            if should_verify:
                logits, verify_logits = model(data.x, data.edge_index, edge_attr=edge_attr, return_verification=True)
            else:
                logits = model(data.x, data.edge_index, edge_attr=edge_attr)
                verify_logits = None
            
            # Extract node capacities (first feature column)
            if data.x.dtype == torch.long:
                node_capacities = data.x[:, 0]
            else:
                node_capacities = data.x[:, 0].long()
            
            # Get edge conflicts for this batch
            edge_conflicts = getattr(data, 'edge_conflicts', None)
            
            # Compute combined loss
            use_aux_losses = loss_weights is not None and (
                loss_weights.get('degree', 0) > 0 or 
                loss_weights.get('crossing', 0) > 0 or
                loss_weights.get('verify', 0) > 0
            )
            
            if use_aux_losses:
                losses = compute_combined_loss(
                    logits, data.y, data.edge_index, node_capacities,
                    edge_conflicts, data.edge_mask, loss_weights,
                    verify_logits=verify_logits,
                    edge_batch=edge_batch
                )
                loss = losses['total']
                
                # Track individual loss components
                total_ce_loss += losses['ce'].item() * data.num_graphs
                total_degree_loss += losses['degree'].item() * data.num_graphs
                total_crossing_loss += losses['crossing'].item() * data.num_graphs
                total_verify_loss += losses['verify'].item() * data.num_graphs
                total_verify_acc += losses['verify_acc'].item()
                if losses['verify'].item() > 0:
                    num_verify_batches += 1
            else:
                # Fallback to standard CE loss
                logits_original = logits[data.edge_mask]
                loss = criterion(logits_original, data.y) if criterion else torch.nn.functional.cross_entropy(logits_original, data.y)
                total_ce_loss += loss.item() * data.num_graphs
            
            # Backward pass (training only)
            if training:
                loss.backward()
                optimizer.step()
            
            # Metrics computation
            logits_original = logits[data.edge_mask]
            total_loss += loss.item() * data.num_graphs
            pred = logits_original.argmax(dim=-1)
            correct_predictions += (pred == data.y).sum().item()
            total_edges += data.edge_mask.sum().item()
            
            # Calculate perfect puzzle accuracy for this batch
            edge_batch_original = edge_batch[data.edge_mask]
            _, num_perfect, num_total = calculate_batch_perfect_puzzles(
                logits_original, data.y,
                torch.ones_like(data.edge_mask[data.edge_mask], dtype=torch.bool),
                edge_batch_original
            )
            perfect_puzzle_stats.append((num_perfect, num_total))
    
    # Compute final metrics
    num_samples = len(loader.dataset)
    metrics = EpochMetrics()
    metrics.loss = total_loss / num_samples
    metrics.ce_loss = total_ce_loss / num_samples
    metrics.degree_loss = total_degree_loss / num_samples if total_degree_loss > 0 else 0.0
    metrics.crossing_loss = total_crossing_loss / num_samples if total_crossing_loss > 0 else 0.0
    metrics.verify_loss = total_verify_loss / num_samples if total_verify_loss > 0 else 0.0
    metrics.verify_accuracy = total_verify_acc / num_verify_batches if num_verify_batches > 0 else 0.0
    metrics.accuracy = correct_predictions / total_edges
    
    # Calculate overall perfect puzzle accuracy
    total_perfect = sum(perfect for perfect, _ in perfect_puzzle_stats)
    total_puzzles = sum(total for _, total in perfect_puzzle_stats)
    metrics.perfect_accuracy = total_perfect / total_puzzles if total_puzzles > 0 else 0.0
    
    return metrics


def main() -> None:
    """Train the Hashi graph model based on the configuration."""
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Train a GNN for Hashi edge classification.")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to the configuration file.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    train_config = config['training']

    # Set device
    device = get_device(train_config['device'])
    print(f"Using device: {device}")

    # Set experiment name
    mlflow.set_experiment("Hashi Graph GNN")

    with mlflow.start_run():
        # Log hyperparameters
        params = flatten_config(config)
        params['config_file'] = args.config
        mlflow.log_params(params)

        # Load datasets
        root_path = Path(data_config['root_dir'])
        size_filter = data_config.get('size') or None
        difficulty_filter = data_config.get('difficulty') or None
        limit = data_config.get('limit') or None

        # Get use_degree and use_meta_node settings from model config
        use_degree = model_config.get('use_degree', False)
        use_meta_node = model_config.get('use_meta_node', False)
        use_row_col_meta = model_config.get('use_row_col_meta', False)
        use_meta_mesh = model_config.get('use_meta_mesh', False)
        use_meta_row_col_edges = model_config.get('use_meta_row_col_edges', False)
        use_distance = model_config.get('use_distance', False)
        use_edge_labels_as_features = model_config.get('use_edge_labels_as_features', False)
        use_closeness_centrality = model_config.get('use_closeness_centrality', False)
        use_conflict_edges = model_config.get('use_conflict_edges', False)
        use_verification_head = model_config.get('use_verification_head', False)

        print("Loading training data...")
        train_dataset = HashiDataset(
            root=root_path, split='train', size=size_filter, difficulty=difficulty_filter, 
            limit=limit, use_degree=use_degree, use_meta_node=use_meta_node,
            use_row_col_meta=use_row_col_meta,
            use_meta_mesh=use_meta_mesh,
            use_meta_row_col_edges=use_meta_row_col_edges,
            use_distance=use_distance,
            use_edge_labels_as_features=use_edge_labels_as_features,
            use_closeness_centrality=use_closeness_centrality,
            use_conflict_edges=use_conflict_edges
        )
        print("Loading validation data...")
        val_dataset = HashiDataset(
            root=root_path, split='val', size=size_filter, difficulty=difficulty_filter, 
            limit=limit, use_degree=use_degree, use_meta_node=use_meta_node,
            use_row_col_meta=use_row_col_meta,
            use_meta_mesh=use_meta_mesh,
            use_meta_row_col_edges=use_meta_row_col_edges,
            use_distance=use_distance,
            use_edge_labels_as_features=use_edge_labels_as_features,
            use_closeness_centrality=use_closeness_centrality,
            use_conflict_edges=use_conflict_edges
        )
        
        # DataLoader setup (num_workers=0 is safest for MPS)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_with_conflicts
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_with_conflicts
        )

        # Initialize model, optimizer, and criterion
        model_type = model_config.get('type', 'gcn').lower()
        
        # Determine edge dimension based on enabled features
        edge_dim = 3  # base: [inv_dx, inv_dy, is_meta]
        if use_conflict_edges:
            edge_dim += 1  # add is_conflict
        if use_meta_mesh:
            edge_dim += 1  # add is_meta_mesh
        if use_meta_row_col_edges:
            edge_dim += 1  # add is_meta_row_col_cross
        if use_edge_labels_as_features:
            edge_dim += 2  # add bridge_label, is_labeled
        
        if model_type == 'gcn':
            model = GCNEdgeClassifier(
                node_embedding_dim=model_config['node_embedding_dim'],
                hidden_channels=model_config['hidden_channels'],
                num_layers=model_config['num_layers'],
                dropout=model_config.get('dropout', 0.25),
                use_degree=use_degree,
                use_meta_node=use_meta_node,
                use_closeness=use_closeness_centrality
                # GCN doesn't support row_col_meta explicitly other than through graph structure
            ).to(device)
        elif model_type == 'gat':
            model = GATEdgeClassifier(
                node_embedding_dim=model_config['node_embedding_dim'],
                hidden_channels=model_config['hidden_channels'],
                num_layers=model_config['num_layers'],
                heads=model_config.get('heads', 8),
                dropout=model_config.get('dropout', 0.25),
                use_degree=use_degree,
                use_meta_node=use_meta_node,
                use_row_col_meta=use_row_col_meta,
                edge_dim=edge_dim,
                use_closeness=use_closeness_centrality
            ).to(device)
        elif model_type == 'gine':
            model = GINEEdgeClassifier(
                node_embedding_dim=model_config['node_embedding_dim'],
                hidden_channels=model_config['hidden_channels'],
                num_layers=model_config['num_layers'],
                dropout=model_config.get('dropout', 0.25),
                use_degree=use_degree,
                use_meta_node=use_meta_node,
                use_row_col_meta=use_row_col_meta,
                edge_dim=edge_dim,
                use_closeness=use_closeness_centrality
            ).to(device)
        elif model_type == 'transformer':
            model = TransformerEdgeClassifier(
                node_embedding_dim=model_config['node_embedding_dim'],
                hidden_channels=model_config['hidden_channels'],
                num_layers=model_config['num_layers'],
                heads=model_config.get('heads', 4),
                dropout=model_config.get('dropout', 0.25),
                use_degree=use_degree,
                use_meta_node=use_meta_node,
                use_row_col_meta=use_row_col_meta,
                edge_dim=edge_dim,
                use_closeness=use_closeness_centrality,
                use_verification_head=use_verification_head
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'gcn', 'gat', 'gine', or 'transformer'.")
        
        print(f"Initialized {model_type.upper()} model")

        # Use datetime for local model save path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
        criterion = torch.nn.CrossEntropyLoss()
        early_stopper = EarlyStopper(
            patience=train_config['early_stopping']['patience'],
            min_delta=train_config['early_stopping']['min_delta']
        )
        
        # Get loss weights from config (verify weight is the max, actual is computed dynamically)
        loss_config = train_config.get('loss_weights', {})
        max_verify_weight = loss_config.get('verify', 0.1)
        min_verify_weight = loss_config.get('min_verify_weight', 0.01)
        base_loss_weights = {
            'ce': loss_config.get('ce', 1.0),
            'degree': loss_config.get('degree', 0.1),
            'crossing': loss_config.get('crossing', 0.5),
            'verify': 0.0  # Will be set dynamically each epoch
        }
        print(f"Using loss weights: CE={base_loss_weights['ce']}, Degree={base_loss_weights['degree']}, "
              f"Crossing={base_loss_weights['crossing']}. Verification (max)={max_verify_weight}, (min)={min_verify_weight}")
        if use_verification_head:
            print(f"Verification head enabled: λ_verify = min({max_verify_weight}, max(min_verify_weight, mask_rate))")

        # Training loop
        print("\nStarting training...")
        best_val_loss = float('inf')
        masking_config = train_config.get('masking', {})
        eval_interval = train_config.get('eval_interval', 1)  # Evaluate every X epochs

        for epoch in range(1, train_config['epochs'] + 1):
            # Calculate current masking rate for training
            current_masking_rate = get_masking_rate(epoch, masking_config, train_config['epochs'])
            
            # Compute dynamic verification weight: min(max_weight, max(min_verify_weight, mask_rate))
            # This ensures a small baseline signal even during warmup, ramping with masking
            current_verify_weight = min(max_verify_weight, max(min_verify_weight, current_masking_rate)) if use_verification_head else 0.0
            current_loss_weights = {**base_loss_weights, 'verify': current_verify_weight}
            
            # Training epoch
            train_metrics = run_epoch(
                model, train_loader, device,
                training=True,
                optimizer=optimizer,
                criterion=criterion,
                masking_rate=current_masking_rate,
                loss_weights=current_loss_weights,
                use_verification=use_verification_head
            )
            
            # Validation with 100% masking to measure true puzzle-solving ability
            if epoch % eval_interval == 0:
                val_metrics = run_epoch(
                    model, val_loader, device,
                    training=False,
                    criterion=criterion,
                    masking_rate=1.0,
                    loss_weights=current_loss_weights,
                    use_verification=use_verification_head
                )

                print(f"Epoch: {epoch:03d}, Train Mask: {current_masking_rate:.2f}, "
                      f"Train Loss: {train_metrics.loss:.4f}, Train Acc: {train_metrics.accuracy:.4f}, Train Perfect: {train_metrics.perfect_accuracy:.4f}, "
                      f"Val Loss: {val_metrics.loss:.4f}, Val Acc: {val_metrics.accuracy:.4f}, Val Perfect: {val_metrics.perfect_accuracy:.4f}")
                
                # Print loss components if auxiliary losses are enabled
                if base_loss_weights['degree'] > 0 or base_loss_weights['crossing'] > 0 or use_verification_head:
                    print(f"  -> Train Loss: CE={train_metrics.ce_loss:.4f}, Degree={train_metrics.degree_loss:.4f}, "
                          f"Crossing={train_metrics.crossing_loss:.4f}, Verify={train_metrics.verify_loss:.4f}")
                    print(f"  -> Val Loss:   CE={val_metrics.ce_loss:.4f}, Degree={val_metrics.degree_loss:.4f}, "
                          f"Crossing={val_metrics.crossing_loss:.4f}, Verify={val_metrics.verify_loss:.4f}")
                    if use_verification_head:
                        print(f"  -> Verify Acc: Train={train_metrics.verify_accuracy:.4f}, Val={val_metrics.verify_accuracy:.4f}, "
                              f"λ_verify={current_verify_weight:.4f}")
                
                # Log metrics
                mlflow.log_metrics({
                    "train_loss": train_metrics.loss,
                    "train_acc": train_metrics.accuracy,
                    "train_perfect_puzzle_acc": train_metrics.perfect_accuracy,
                    "train_ce_loss": train_metrics.ce_loss,
                    "train_degree_loss": train_metrics.degree_loss,
                    "train_crossing_loss": train_metrics.crossing_loss,
                    "train_verify_loss": train_metrics.verify_loss,
                    "train_verify_acc": train_metrics.verify_accuracy,
                    "val_loss": val_metrics.loss,
                    "val_acc": val_metrics.accuracy,
                    "val_perfect_puzzle_acc": val_metrics.perfect_accuracy,
                    "val_ce_loss": val_metrics.ce_loss,
                    "val_degree_loss": val_metrics.degree_loss,
                    "val_crossing_loss": val_metrics.crossing_loss,
                    "val_verify_loss": val_metrics.verify_loss,
                    "val_verify_acc": val_metrics.verify_accuracy,
                    "masking_rate": current_masking_rate,
                    "verify_weight": current_verify_weight
                }, step=epoch)

                # Save the best model and early stopping only if evaluation was performed
                if val_metrics.loss < best_val_loss:
                    best_val_loss = val_metrics.loss
                    model_save_path = Path("models") / f"best_model_{timestamp}.pt"
                    model_save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)
                    print(f"  -> New best model saved to {model_save_path}")
                    mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)

                # Early stopping
                if early_stopper.early_stop(val_metrics.loss):
                    print("Early stopping triggered.")
                    break
            else:
                # Log only training metrics if validation was skipped
                mlflow.log_metrics({
                    "train_loss": train_metrics.loss,
                    "train_acc": train_metrics.accuracy,
                    "train_perfect_puzzle_acc": train_metrics.perfect_accuracy,
                    "train_ce_loss": train_metrics.ce_loss,
                    "train_degree_loss": train_metrics.degree_loss,
                    "train_crossing_loss": train_metrics.crossing_loss,
                    "train_verify_loss": train_metrics.verify_loss,
                    "train_verify_acc": train_metrics.verify_accuracy,
                    "masking_rate": current_masking_rate,
                    "verify_weight": current_verify_weight
                }, step=epoch)


if __name__ == "__main__":
    main()
