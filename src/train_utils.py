"""
Utility functions for training and evaluation metrics.
"""
import torch
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from typing import Dict


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
    
    # Per-edge incorrectness: 1 if wrong, 0 if correct
    edge_incorrect = (masked_preds != masked_targets).long()
    
    # Sum errors per puzzle using scatter
    num_puzzles = masked_batch.max().item() + 1
    errors_per_puzzle = scatter(edge_incorrect, masked_batch, dim=0,
                                 dim_size=num_puzzles, reduce='sum')
    
    # Perfect puzzles have 0 errors
    perfect_mask = errors_per_puzzle == 0
    perfect_puzzles = perfect_mask.sum().item()
    
    perfect_accuracy = perfect_puzzles / num_puzzles if num_puzzles > 0 else 0.0
    return perfect_accuracy, perfect_puzzles, num_puzzles


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

