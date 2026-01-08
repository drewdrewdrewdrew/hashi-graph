"""
Shared utilities for Hashi GNN.
"""
import torch
import yaml
from typing import Dict, Any, List, Tuple

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_puzzle_solved(
    current_bridges: torch.Tensor,
    data: Any,
    model_config: Dict[str, Any]
) -> bool:
    """
    Check if a puzzle is solved given current bridge state.

    Args:
        current_bridges: Current bridge counts [num_edges]
        data: PyG Data object with puzzle information
        model_config: Model configuration

    Returns:
        bool: True if puzzle is solved (all constraints satisfied)
    """
    # Get node capacities
    node_capacities = data.node_type.float()  # Assume node_type contains capacities

    # Only check puzzle nodes (not meta nodes)
    is_puzzle_node = (data.node_type > 0) & (data.node_type <= 8)  # Islands are 1-8

    # Calculate current degree for each node
    row, col = data.edge_index
    degree = torch.zeros(data.x.size(0), dtype=current_bridges.dtype, device=current_bridges.device)
    degree.scatter_add_(0, row, current_bridges)
    degree.scatter_add_(0, col, current_bridges)

    # Check degree constraints for puzzle nodes
    puzzle_degrees = degree[is_puzzle_node]
    puzzle_capacities = node_capacities[is_puzzle_node]

    # All puzzle nodes must have degree equal to capacity
    degree_satisfied = torch.all(puzzle_degrees == puzzle_capacities)

    # Check bridge constraints (no more than 2 bridges per edge)
    bridge_satisfied = torch.all(current_bridges <= 2)

    # Check no crossing violations (simplified - assume bridges are properly placed)
    # In a full implementation, this would check for invalid crossings

    return degree_satisfied.item() and bridge_satisfied.item()


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

def clear_memory_cache(device: torch.device) -> None:
    """Clear GPU/MPS memory cache to prevent fragmentation."""
    if device.type == 'mps':
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def _normalize_conflict_index(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return _normalize_conflict_index(value[0])
    return int(value)

def _normalize_conflict_pair(conflict: Any) -> Tuple[int, int]:
    if isinstance(conflict, torch.Tensor):
        conflict = conflict.tolist()
    if not isinstance(conflict, (list, tuple)) or len(conflict) != 2:
        raise ValueError("Each edge_conflict must contain exactly two entries.")
    return _normalize_conflict_index(conflict[0]), _normalize_conflict_index(conflict[1])

from torch_geometric.data import Batch, Data

def custom_collate_with_conflicts(data_list: List[Data]) -> Batch:
    """
    Custom collate function that properly handles edge_conflicts during batching.
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
            graph_conflicts = []
            for conflict in conflicts:
                e1, e2 = _normalize_conflict_pair(conflict)
                graph_conflicts.append((e1, e2))

            if graph_conflicts:
                conflict_tensor = torch.tensor(graph_conflicts, dtype=torch.long)
                offset_tensor = torch.tensor([edge_offset, edge_offset], dtype=torch.long)
                offset_conflicts = conflict_tensor + offset_tensor
                all_conflicts.extend(offset_conflicts.tolist())

        edge_offset += data.edge_index.size(1)
    
    batch.edge_conflicts = all_conflicts if all_conflicts else None
    return batch

