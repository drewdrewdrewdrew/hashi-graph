"""
Core training engine for Hashi GNN.
Centralizes model creation, dataset loading, and training loop components.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch_geometric.data import Data
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List

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

class Trainer:
    """
    Encapsulates training logic, providing a unified interface for model
    initialization, dataloader creation, and epoch execution.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device, callbacks: List[Any] = []):
        self.config = config
        self.device = device
        self.callbacks = callbacks
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.current_masking_rate = 0.0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')

    def _setup(self, train_transform: Optional[Any] = None):
        """Internal setup for model, optimizer, and data loaders."""
        self.model = self.create_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate']
        )
        self.train_loader = self.create_dataloader(split='train', transform=train_transform)
        self.val_loader = self.create_dataloader(split='val')

    def train(self, train_transform: Optional[Any] = None):
        """Main training loop."""
        self._setup(train_transform)
        
        epochs = self.config['training']['epochs']
        eval_interval = self.config['training'].get('eval_interval', 1)
        accumulation_steps = self.config['training'].get('accumulation_steps', 1)
        early_stopping_config = self.config['training'].get('early_stopping', {})
        early_stopper = EarlyStopper(
            patience=early_stopping_config.get('patience', 10),
            min_delta=early_stopping_config.get('min_delta', 0.0)
        )

        for callback in self.callbacks:
            callback.on_train_start(self)

        try:
            for epoch in range(1, epochs + 1):
                for callback in self.callbacks:
                    callback.on_epoch_start(self, epoch)

                # Check if AR mode is enabled
                ar_mode = self.config['training'].get('ar_mode', False)

                if ar_mode:
                    # AR training with rollouts
                    ar_steps = self.config['training'].get('ar_steps', 5)
                    head_type = self.config['model'].get('head_type', 'regression')
                    overshoot_penalty = self.config['training'].get('overshoot_penalty', 5.0)

                    train_metrics = run_ar_epoch(
                        self.model, self.train_loader, ar_steps=ar_steps,
                        head_type=head_type, overshoot_penalty=overshoot_penalty,
                        training=True, optimizer=self.optimizer,
                        accumulation_steps=accumulation_steps, model_config=self.config['model']
                    )
                else:
                    # Standard One-Shot training
                    self.current_masking_rate = get_masking_rate(epoch, self.config['training']['masking'], epochs)

                    train_metrics = self.run_epoch(
                        self.model, self.train_loader, training=True,
                        optimizer=self.optimizer, masking_rate=self.current_masking_rate,
                        accumulation_steps=accumulation_steps
                    )
                
                # Clear memory after training pass
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
                elif self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                val_metrics = None
                if epoch % eval_interval == 0:
                    if ar_mode:
                        # AR validation (5-step rollouts as proxy for full solve)
                        val_metrics = run_ar_epoch(
                            self.model, self.val_loader, ar_steps=ar_steps,
                            head_type=head_type, overshoot_penalty=overshoot_penalty,
                            training=False, model_config=self.config['model']
                        )
                    else:
                        # Standard One-Shot validation
                        val_metrics = self.run_epoch(
                            self.model, self.val_loader, training=False, masking_rate=1.0
                        )
                    # Clear memory after validation pass
                    if self.device.type == 'mps':
                        torch.mps.empty_cache()
                    elif self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch, train_metrics, val_metrics)

                if val_metrics:
                    self.best_val_acc = max(self.best_val_acc, val_metrics.accuracy)
                    self.best_val_loss = min(self.best_val_loss, val_metrics.loss)
                    
                    if early_stopper.early_stop(val_metrics.loss):
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
        finally:
            for callback in self.callbacks:
                callback.on_train_end(self)

    def create_model(self) -> torch.nn.Module:
        """Create and return the model based on config."""
        model_config = self.config['model']
        model_type = model_config.get('type', 'gcn').lower()
        
        edge_dim = self.compute_edge_dim()
        
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
            'use_articulation_points': model_config.get('use_articulation_points', False),
            'use_spectral_features': model_config.get('use_spectral_features', False),
        }

        if model_type == 'gcn':
            # GCN might accept kwargs or just ignore them if we updated it to accept **kwargs
            # For now, GCN only accepts what it defines. We should update GCN if we want to support new features there.
            # But we are focusing on Transformer.
            # To be safe, we can filter or just assume model accepts **kwargs if we updated it.
            # I didn't update GCN/GAT/GINE yet.
            # So I will pass specific args if model_type is transformer, or be careful.
            # Actually, Python classes without **kwargs will crash if unexpected args are passed.
            # TransformerEdgeClassifier has **kwargs now.
            model = GCNEdgeClassifier(**common_kwargs) # This might crash if not updated!
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
                edge_concat_global_meta=model_config.get('edge_concat_global_meta', False),
                head_type=model_config.get('head_type', 'classification')
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model.to(self.device)

    def compute_edge_dim(self) -> int:
        """Calculate edge dimension based on enabled features."""
        model_config = self.config['model']
        edge_dim = 3  # base: [inv_dx, inv_dy, is_meta]
        if model_config.get('use_conflict_edges', False):
            edge_dim += 1
        if model_config.get('use_meta_mesh', False):
            edge_dim += 1
        if model_config.get('use_meta_row_col_edges', False):
            edge_dim += 1
        if model_config.get('use_edge_labels_as_features', False):
            edge_dim += 2
        if model_config.get('use_cut_edges', False):
            edge_dim += 1
        return edge_dim

    def create_dataloader(
        self, 
        split: str, 
        transform: Optional[Any] = None,
        use_cache: bool = False
    ) -> DataLoader:
        """Create a dataloader for the specified split."""
        if use_cache:
            dataset = HashiDatasetCache.get_or_create(self.config, split, transform=transform)
        else:
            data_config = self.config['data']
            model_config = self.config['model']
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
                use_articulation_points=model_config.get('use_articulation_points', False),
                use_cut_edges=model_config.get('use_cut_edges', False),
                use_spectral_features=model_config.get('use_spectral_features', False),
                transform=transform
            )

        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=(split == 'train'),
            num_workers=self.config['training'].get('num_workers', 0),
            collate_fn=custom_collate_with_conflicts,
            persistent_workers=self.config['training'].get('use_persistent_workers', False)
        )

    def run_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        training: bool = True,
        optimizer: Optional[Optimizer] = None,
        masking_rate: float = 0.0,
        accumulation_steps: int = 1,
    ) -> EpochMetrics:
        """Execute a single epoch of training or evaluation."""
        if training:
            if optimizer is None:
                raise ValueError("Optimizer required for training mode")
            model.train()
            desc = "Training"
        else:
            model.eval()
            desc = "Evaluating"
        
        total_loss = torch.tensor(0.0, device=self.device)
        total_ce_loss = torch.tensor(0.0, device=self.device)
        total_degree_loss = torch.tensor(0.0, device=self.device)
        total_crossing_loss = torch.tensor(0.0, device=self.device)
        total_verify_loss = torch.tensor(0.0, device=self.device)
        total_verify_acc = torch.tensor(0.0, device=self.device)
        total_verify_recall_pos = torch.tensor(0.0, device=self.device)
        total_verify_recall_neg = torch.tensor(0.0, device=self.device)
        correct_predictions = torch.tensor(0, device=self.device)
        total_edges = torch.tensor(0, device=self.device)
        perfect_puzzle_stats = []
        num_verify_batches = 0
        
        loss_weights = self.config['training'].get('loss_weights')
        use_verification = self.config['model'].get('use_verification_head', False)

        context = torch.no_grad() if not training else torch.enable_grad()
        
        with context:
            if training:
                optimizer.zero_grad()
            
            for batch_idx, data in enumerate(tqdm(loader, desc=desc, leave=False)):
                data = data.to(self.device)
                
                # Always run masking logic to ensure unused_capacity is synced 
                # with the current masking_rate (even at 0.0 or 1.0)
                data = apply_edge_label_masking(data, masking_rate, self.device, self.config)
                
                edge_attr = getattr(data, 'edge_attr', None)
                edge_batch = get_edge_batch_indices(data)
                node_type = getattr(data, 'node_type', None)

                model_has_verify = hasattr(model, 'use_verification_head') and model.use_verification_head
                should_verify = use_verification and model_has_verify

                if should_verify:
                    logits, verify_logits = model(
                        data.x, data.edge_index, edge_attr=edge_attr,
                        batch=getattr(data, 'batch', None), node_type=node_type, return_verification=True
                    )
                else:
                    logits = model(
                        data.x, data.edge_index, edge_attr=edge_attr,
                        batch=getattr(data, 'batch', None), node_type=node_type
                    )
                    verify_logits = None

                # Use node_type for capacities if available, otherwise fall back to x[:, 0]
                node_capacities = node_type if node_type is not None else data.x[:, 0].long()
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
                
                # Masking for accurate accuracy metrics
                logits_original = logits[data.edge_mask]
                total_loss += loss * data.num_graphs
                pred = logits_original.argmax(dim=-1)
                correct_predictions += (pred == data.y).sum()
                total_edges += data.edge_mask.sum()
                
                edge_batch_original = edge_batch[data.edge_mask]
                # Fix: Pass correct mask for accuracy calculation (filtered original edges)
                accuracy_mask = torch.ones(logits_original.size(0), dtype=torch.bool, device=self.device)
                _, num_perfect, num_total = calculate_batch_perfect_puzzles(
                    logits_original, data.y,
                    accuracy_mask,
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
    if data.edge_attr is None:
        return data

    edge_dim = data.edge_attr.size(1)
    if edge_dim < 2:
        return data

    # Dynamic indices based on config
    # We need to find where bridge_label and is_labeled are
    # Order in data.py:
    # ...
    # - bridge_label, is_labeled (if use_edge_labels_as_features)
    # - is_cut_edge (if use_cut_edges)
    
    # We must calculate indices dynamically
    model_config = config.get('model', {})
    
    # Calculate index offset
    # Base: inv_dx, inv_dy, is_meta (3)
    current_idx = 3
    if model_config.get('use_conflict_edges', False): current_idx += 1
    if model_config.get('use_meta_mesh', False): current_idx += 1
    if model_config.get('use_meta_row_col_edges', False): current_idx += 1
    
    # Now we are at bridge_label
    if not model_config.get('use_edge_labels_as_features', False):
        return data # Can't mask if features don't exist
        
    bridge_label_idx = current_idx
    is_labeled_idx = current_idx + 1
    # is_cut_edge would be at current_idx + 2

    # Verify edge dim is large enough
    if edge_dim <= is_labeled_idx:
        return data

    use_capacity = model_config.get('use_capacity', True)
    use_structural_degree = model_config.get('use_structural_degree', True)
    use_structural_degree_nsew = model_config.get('use_structural_degree_nsew', False)
    use_unused_capacity = model_config.get('use_unused_capacity', True)

    unused_capacity_idx = 0
    if use_capacity: unused_capacity_idx += 1
    if use_structural_degree or use_structural_degree_nsew: unused_capacity_idx += 1

    # Fix: Always clone x if we might modify it or if we need to reset unused capacity
    if use_unused_capacity:
        data.x = data.x.clone()
        # Reset unused capacity to 0 (assuming state where we only count MASKED edges as unused)
        # This fixes the bug where we added masked capacity to already full capacity
        data.x[:, unused_capacity_idx] = 0.0

    if masking_rate <= 0.0:
        return data

    data.edge_attr = data.edge_attr.clone()

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


def run_ar_epoch(
    model,
    loader: DataLoader,
    ar_steps: int = 5,
    head_type: str = 'regression',
    overshoot_penalty: float = 5.0,
    training: bool = True,
    optimizer: Optional[Optimizer] = None,
    accumulation_steps: int = 1,
    model_config: Optional[Dict[str, Any]] = None,
) -> EpochMetrics:
    """
    Execute an AR (Incremental) training epoch with rollouts.

    Args:
        model: The model to train/evaluate
        loader: DataLoader with puzzle batches
        ar_steps: Number of steps per rollout
        head_type: 'regression' or 'conditional'
        overshoot_penalty: Penalty for overshooting in regression mode
        training: Whether to train (True) or evaluate (False)
        optimizer: Optimizer for training
        accumulation_steps: Gradient accumulation steps

    Returns:
        EpochMetrics with loss and accuracy
    """
    from .train_utils import update_node_features, select_ar_action, apply_ar_action
    from .losses import compute_ar_loss
    from .models.heads import RegressionActionHead, ConditionalActionHead

    if training and optimizer is None:
        raise ValueError("Optimizer required for training mode")

    model.train() if training else model.eval()

    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    total_rollouts = 0

    context = torch.enable_grad if training else torch.no_grad

    with context():
        if training:
            optimizer.zero_grad()

        for batch_idx, data in enumerate(tqdm(loader, desc="AR Training" if training else "AR Evaluating", leave=False)):
            data = data.to(next(model.parameters()).device)

            # Initialize random starting state for rollout
            current_bridges = initialize_random_bridge_state(data.y, data.edge_mask)

            rollout_loss = 0.0

            # Perform rollout for ar_steps
            for step in range(ar_steps):
                # Update node features based on current bridge state
                data.x = update_node_features(
                    data.x, current_bridges, data.edge_index,
                    data.node_type, model_config
                )

                # Forward pass
                edge_attr = getattr(data, 'edge_attr', None)
                output = model(data.x, data.edge_index, edge_attr=edge_attr)

                # Compute loss on valid edges
                targets = data.y - current_bridges  # Remaining capacity
                step_loss = compute_ar_loss(
                    output, targets, current_bridges, data.edge_mask,
                    head_type=head_type, overshoot_penalty=overshoot_penalty
                )

                if training:
                    # Scale loss for accumulation
                    step_loss = step_loss / (ar_steps * accumulation_steps)
                    step_loss.backward()

                rollout_loss += step_loss.item()

                # Select action (inference-like)
                edge_idx, confidence = select_ar_action(
                    output, current_bridges, data.edge_mask, head_type
                )

                if edge_idx == -1:
                    # No valid actions - end rollout early
                    break

                # Apply action with minimal hint correction
                predicted_action = get_action_from_output(output[edge_idx:edge_idx+1], head_type)
                correct_action = targets[edge_idx].item()

                if predicted_action != correct_action:
                    # Apply minimal hint: move one step closer to truth
                    applied_action = min(correct_action, predicted_action + 1)
                else:
                    applied_action = predicted_action

                current_bridges = apply_ar_action(current_bridges, applied_action, edge_idx)

            # Accumulate gradients after rollout
            if training and (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Add to total loss (rollout_loss is already accumulated per step)
            total_loss += rollout_loss
            total_rollouts += data.num_graphs if hasattr(data, 'num_graphs') else 1

    # Return metrics
    metrics = EpochMetrics()
    metrics.loss = (total_loss / total_rollouts).item() if total_rollouts > 0 else 0.0
    # For now, we'll focus on loss. Accuracy calculation for AR is more complex
    # and can be added later when we have evaluation metrics
    metrics.accuracy = 0.0  # Placeholder
    metrics.perfect_accuracy = 0.0  # Placeholder

    return metrics


def initialize_random_bridge_state(final_bridges: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
    """
    Initialize a random partial bridge state for AR training.

    This creates a "starting point" for rollouts by randomly selecting some bridges
    to already be placed, simulating different points in the solving process.

    Args:
        final_bridges: Final bridge counts [num_edges]
        edge_mask: Boolean mask for valid edges [num_edges]

    Returns:
        Random partial bridge state [num_edges]
    """
    # Start with all zeros
    current_bridges = torch.zeros_like(final_bridges)

    # For each valid edge, randomly decide how many bridges to "pre-place"
    for i in range(len(final_bridges)):
        if not edge_mask[i]:
            continue

        final_count = final_bridges[i].item()
        if final_count == 0:
            continue  # No bridges to place

        # Randomly choose how many bridges are already placed (0 to final_count)
        pre_placed = torch.randint(0, int(final_count) + 1, (1,)).item()
        current_bridges[i] = pre_placed

    return current_bridges


def get_action_from_output(output: torch.Tensor, head_type: str) -> int:
    """Extract discrete action from model output."""
    from .models.heads import RegressionActionHead, ConditionalActionHead

    if head_type == 'regression':
        action, _ = RegressionActionHead.predict_action_static(output)
        return action
    else:  # conditional
        action, _ = ConditionalActionHead.predict_action_static(output)
        return action
