"""Shared utilities for Hashi GNN."""

import pathlib
from typing import Any

import torch
from torch_geometric.data import Batch, Data
import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    with pathlib.Path(config_path).open() as f:
        return yaml.safe_load(f)


def check_puzzle_solved(
    current_bridges: torch.Tensor,
    data: Data,
    _model_config: dict[str, Any],
) -> bool:
    """
    Check if a puzzle is solved given current bridge state.

    Args:
        current_bridges: Current bridge counts [num_edges]
        data: PyG Data object with puzzle information
        model_config: Model configuration

    Returns
    -------
        bool: True if puzzle is solved (all constraints satisfied)
    """
    # Get node capacities
    node_capacities = data.node_type.float()  # Assume node_type contains capacities

    # Only check puzzle nodes (not meta nodes)
    is_puzzle_node = (data.node_type > 0) & (data.node_type <= 8)  # Islands are 1-8

    # Calculate current degree for each node
    row, col = data.edge_index
    degree = torch.zeros(
        data.x.size(0), dtype=current_bridges.dtype, device=current_bridges.device,
    )
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


def flatten_config(
    config: dict[str, Any], parent_key: str = "", sep: str = ".",
) -> dict[str, Any]:
    """Flatten a nested dictionary config."""
    items: list[tuple[str, Any]] = []
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
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_config)


def clear_memory_cache(device: torch.device) -> None:
    """Clear GPU/MPS memory cache to prevent fragmentation."""
    if device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _normalize_conflict_index(
    value: torch.Tensor | list | tuple | int | float,
) -> int:
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return _normalize_conflict_index(value[0])
    return int(value)


def _normalize_conflict_pair(
    conflict: torch.Tensor | list | tuple,
) -> tuple[int, int]:
    if isinstance(conflict, torch.Tensor):
        conflict = conflict.tolist()
    if not isinstance(conflict, (list, tuple)) or len(conflict) != 2:
        error_msg = "Each edge_conflict must contain exactly two entries."
        raise ValueError(error_msg)
    return _normalize_conflict_index(conflict[0]), _normalize_conflict_index(
        conflict[1],
    )


def custom_collate_with_conflicts(data_list: list[Data]) -> Batch:
    """Properly handle edge_conflicts during batching."""
    # We want edge_conflict_index to be a tensor attribute that PyG can slice.
    # We first ensure each data object has its edge_conflicts converted to a tensor.
    for data in data_list:
        # Prefer existing edge_conflict_index tensor if it's non-empty
        if (
            hasattr(data, "edge_conflict_index")
            and isinstance(data.edge_conflict_index, torch.Tensor)
            and data.edge_conflict_index.size(1) > 0
        ):
            continue

        conflicts = getattr(data, "edge_conflicts", [])
        if conflicts is None:
            conflicts = []

        if isinstance(conflicts, torch.Tensor):
            if conflicts.dim() == 2 and conflicts.size(0) != 2 and conflicts.size(1) == 2:
                data.edge_conflict_index = conflicts.t().contiguous()
            else:
                data.edge_conflict_index = conflicts
        elif conflicts:
            # Convert list of tuples to [2, num_conflicts] tensor
            normalized = [_normalize_conflict_pair(c) for c in conflicts]
            data.edge_conflict_index = (
                torch.tensor(normalized, dtype=torch.long).t().contiguous()
            )
        else:
            data.edge_conflict_index = torch.empty((2, 0), dtype=torch.long)

    # Use PyG's standard batching, but we handle edge_conflict_index manually
    # to ensure it's incremented by num_edges and has correct slice metadata.
    batch = Batch.from_data_list(
        data_list,
        exclude_keys=["edge_conflicts", "edge_conflict_index"],
    )

    all_conflicts = []
    slices = [0]
    edge_offsets = []
    edge_offset = 0

    for data in data_list:
        conflicts = data.edge_conflict_index
        if conflicts.size(1) > 0:
            offset_conflicts = conflicts + edge_offset
            all_conflicts.append(offset_conflicts)

        edge_offsets.append(edge_offset)
        edge_offset += data.edge_index.size(1)
        slices.append(slices[-1] + conflicts.size(1))

    if all_conflicts:
        batch.edge_conflict_index = torch.cat(all_conflicts, dim=1)
    else:
        batch.edge_conflict_index = torch.empty((2, 0), dtype=torch.long)

    # Set up slice metadata for Batch[indices] support
    batch._slice_dict["edge_conflict_index"] = torch.tensor(slices, dtype=torch.long)
    batch._inc_dict["edge_conflict_index"] = torch.tensor(
        edge_offsets, dtype=torch.long
    )

    batch.edge_conflicts = None
    return batch
