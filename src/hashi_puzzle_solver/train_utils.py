"""
Utility functions for training and evaluation metrics.
"""
import torch
import yaml
import os
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from typing import Dict, Any, Tuple

# Import action heads for AR functionality
from .models.heads import RegressionActionHead, ConditionalActionHead


def is_puzzle_perfect(data, predictions):
    """
    Check if a single puzzle is perfectly solved.
    
    Args:
        data: PyG Data object for a single puzzle
        predictions: Predicted edge labels [num_edges]
    
    Returns:
        bool: True if all original edges are correctly predicted
    """
    # Only check original puzzle edges (not meta edges)
    original_mask = data.edge_mask if hasattr(data, 'edge_mask') else torch.ones(len(predictions), dtype=torch.bool)
    original_preds = predictions[original_mask]
    original_targets = data.y[original_mask] if hasattr(data, 'edge_mask') else data.y
    
    return torch.all(original_preds == original_targets).item()


def evaluate_puzzle(model: torch.nn.Module, data: Data, device: torch.device) -> Dict:
    """
    Evaluate a single puzzle and return predictions + correctness.
    
    Args:
        model: Trained model
        data: PyG Data object for a single puzzle
        device: torch device
    
    Returns:
        dict: Contains 'predictions', 'targets', 'is_perfect', 'accuracy'
    """
    data = data.to(device)
    model.eval()
    
    with torch.no_grad():
        edge_attr = getattr(data, 'edge_attr', None)
        logits = model(data.x, data.edge_index, edge_attr=edge_attr)
        
        # Get predictions for original edges only
        original_mask = data.edge_mask if hasattr(data, 'edge_mask') else torch.ones(logits.size(0), dtype=torch.bool)
        logits_original = logits[original_mask]
        targets = data.y[original_mask] if hasattr(data, 'edge_mask') else data.y
        predictions = logits_original.argmax(dim=-1)
        
        is_perfect = torch.all(predictions == targets).item()
        accuracy = (predictions == targets).float().mean().item()
        
    return {
        'predictions': predictions.cpu(),
        'targets': targets.cpu(),
        'is_perfect': is_perfect,
        'accuracy': accuracy,
        'num_edges': len(targets)
    }


def calculate_perfect_puzzle_accuracy(predictions, targets, edge_masks, batch_indices):
    """
    Calculate the percentage of puzzles that are 100% correctly solved.
    Uses vectorized scatter operations for efficiency.
    
    Args:
        predictions: Predicted edge labels [num_edges]
        targets: Ground truth edge labels [num_edges]
        edge_masks: Boolean mask for original puzzle edges [num_edges]
        batch_indices: Batch index for each edge [num_edges]
    
    Returns:
        float: Percentage of puzzles with all edges correctly predicted (0.0 to 1.0)
        int: Number of perfect puzzles
        int: Total number of puzzles
    """
    # Apply edge mask to get only original puzzle edges
    masked_preds = predictions[edge_masks]
    masked_targets = targets[edge_masks]
    masked_batch = batch_indices[edge_masks]

    # Handle empty case
    if len(masked_batch) == 0:
        return 0.0, 0, 0

    # Per-edge incorrectness: 1 if wrong, 0 if correct
    edge_incorrect = (masked_preds != masked_targets).long()

    # Sum errors per puzzle using scatter
    num_puzzles = masked_batch.max() + 1
    errors_per_puzzle = scatter(edge_incorrect, masked_batch, dim=0,
                                 dim_size=num_puzzles, reduce='sum')

    # Perfect puzzles have 0 errors
    perfect_mask = errors_per_puzzle == 0
    perfect_puzzles = perfect_mask.sum()

    # Convert to scalars only when returning
    perfect_accuracy = (perfect_puzzles / num_puzzles).item() if num_puzzles > 0 else 0.0
    return perfect_accuracy, perfect_puzzles.item(), num_puzzles.item()


def calculate_batch_perfect_puzzles(logits, targets, edge_masks, batch_indices):
    """
    Calculate perfect puzzle metrics for a batch.
    
    Args:
        logits: Model output logits [num_edges, num_classes]
        targets: Ground truth edge labels [num_edges]
        edge_masks: Boolean mask for original puzzle edges [num_edges]
        batch_indices: Batch index for each edge [num_edges]
    
    Returns:
        tuple: (perfect_accuracy, num_perfect, num_total)
    """
    predictions = logits.argmax(dim=-1)
    return calculate_perfect_puzzle_accuracy(predictions, targets, edge_masks, batch_indices)


def aggregate_perfect_puzzle_stats(stats_list):
    """
    Aggregate perfect puzzle statistics from multiple batches.
    
    Args:
        stats_list: List of tuples (num_perfect, num_total) from each batch
    
    Returns:
        float: Overall percentage of perfect puzzles
        int: Total number of perfect puzzles
        int: Total number of puzzles
    """
    total_perfect = sum(perfect for perfect, _ in stats_list)
    total_puzzles = sum(total for _, total in stats_list)
    
    perfect_accuracy = total_perfect / total_puzzles if total_puzzles > 0 else 0.0
    return perfect_accuracy, total_perfect, total_puzzles


def get_edge_batch_indices(data):
    """
    Get the batch index for each edge in a PyG Batch object.

    Args:
        data: PyG Batch object

    Returns:
        torch.Tensor: Batch index for each edge [num_edges]
    """
    # PyG Batch stores batch indices for nodes in data.batch
    # We need to map edges to their batch indices via their source nodes
    edge_batch = data.batch[data.edge_index[0]]
    return edge_batch


def save_config_to_model_dir(config: Dict[str, Any], model_save_path: str, config_filename: str = "config.yaml"):
    """
    Save a copy of the training config to the model directory.

    This ensures that the exact configuration used to train a model
    is preserved alongside the model weights.

    Args:
        config: The configuration dictionary
        model_save_path: Path where the model will be saved (e.g., "models/best_model_20241220_120000.pt")
        config_filename: Name of the config file (default: "config.yaml")
    """
    # Get the directory where the model will be saved
    model_dir = Path(model_save_path).parent

    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save config as YAML
    config_path = model_dir / config_filename
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Saved config to {config_path}")


def get_unused_capacity_index(model_config: Dict[str, Any]) -> int:
    """
    Get the index of unused_capacity in node feature vector.

    Args:
        model_config: Model configuration dictionary

    Returns:
        Index of unused_capacity feature in node feature vector
    """
    idx = 0

    # Node type is always first (but not counted in features)
    # Features start after node type

    if model_config.get('use_capacity', True):
        idx += 1

    if model_config.get('use_structural_degree', True) or model_config.get('use_structural_degree_nsew', False):
        idx += 1

    # unused_capacity should be at this index
    return idx


def update_node_features(x: torch.Tensor, current_bridges: torch.Tensor,
                        edge_index: torch.Tensor, node_type: torch.Tensor,
                        model_config: Dict[str, Any]) -> torch.Tensor:
    """
    Update node features based on current bridge state.

    Specifically updates the unused_capacity feature to reflect remaining capacity
    after accounting for current bridges.

    Args:
        x: Node features [num_nodes, num_features]
        current_bridges: Current bridge counts [num_edges]
        edge_index: Graph connectivity [2, num_edges]
        node_type: Node types [num_nodes] (1-8 for islands, 9+ for meta)
        model_config: Model configuration

    Returns:
        Updated node features [num_nodes, num_features]
    """
    if not model_config.get('use_unused_capacity', True):
        return x  # No unused_capacity feature to update

    # Create a copy to avoid modifying the original
    x_updated = x.clone()

    # Get the index of unused_capacity feature
    unused_idx = get_unused_capacity_index(model_config)

    # Calculate current degree for each node (sum of bridges on incident edges)
    row, col = edge_index
    # Only count bridges on original edges (not meta edges)
    # We need to identify which edges are original vs meta
    # For now, assume all edges in edge_index are original puzzle edges
    # This might need refinement if meta edges are included

    # Compute degree: sum of current_bridges for edges incident to each node
    degree = torch.zeros(x.size(0), dtype=current_bridges.dtype, device=x.device)
    degree.scatter_add_(0, row, current_bridges)
    degree.scatter_add_(0, col, current_bridges)

    # Update unused_capacity: original_capacity - current_degree
    # For meta nodes (node_type > 8), unused_capacity should remain 0
    is_puzzle_node = (node_type <= 8) & (node_type > 0)  # Islands are 1-8
    original_capacity = x[:, unused_idx]  # Original unused_capacity values
    updated_capacity = torch.where(
        is_puzzle_node,
        original_capacity - degree,  # Subtract used bridges
        original_capacity  # Keep meta nodes as 0
    )

    # Ensure non-negative (clamp to 0)
    updated_capacity = torch.clamp(updated_capacity, min=0.0)

    x_updated[:, unused_idx] = updated_capacity

    return x_updated


def apply_ar_action(current_bridges: torch.Tensor, action: int, edge_idx: int) -> torch.Tensor:
    """
    Apply an AR action to the current bridge state.

    Args:
        current_bridges: Current bridge counts [num_edges]
        action: Action to apply (0, 1, or 2 bridges to add)
        edge_idx: Index of edge to modify

    Returns:
        Updated bridge state [num_edges]
    """
    new_bridges = current_bridges.clone()
    new_bridges[edge_idx] += action

    # Ensure we don't exceed max bridges (2)
    new_bridges[edge_idx] = torch.clamp(new_bridges[edge_idx], max=2.0)

    return new_bridges


def select_ar_action(output: torch.Tensor, current_bridges: torch.Tensor,
                    edge_mask: torch.Tensor, head_type: str = 'regression') -> Tuple[int, float]:
    """
    Select the best AR action from model output.

    Args:
        output: Model output (shape depends on head_type)
        current_bridges: Current bridge counts [num_edges]
        edge_mask: Boolean mask for valid edges [num_edges]
        head_type: 'regression' or 'conditional'

    Returns:
        Tuple of (edge_idx, confidence) for best action
    """
    # Get predictions for each edge
    if head_type == 'regression':
        # output shape: [num_edges]
        predictions = []
        confidences = []
        for i in range(len(output)):
            if not edge_mask[i]:
                continue  # Skip masked edges
            if current_bridges[i] >= 2:
                continue  # Skip locked edges

            action, confidence = RegressionActionHead.predict_action_static(output[i:i+1])
            if action > 0:  # Only consider constructive actions
                predictions.append((i, action, confidence))
    else:  # conditional
        # output shape: [num_edges, 2]
        predictions = []
        for i in range(len(output)):
            if not edge_mask[i]:
                continue  # Skip masked edges
            if current_bridges[i] >= 2:
                continue  # Skip locked edges

            action, confidence = ConditionalActionHead.predict_action_static(output[i])
            if action > 0:  # Only consider constructive actions
                predictions.append((i, action, confidence))

    if not predictions:
        # No valid actions available
        return -1, 0.0

    # Select action with highest confidence
    best_edge, best_action, best_confidence = max(predictions, key=lambda x: x[2])

    return best_edge, best_confidence

