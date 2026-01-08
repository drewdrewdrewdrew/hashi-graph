"""
AR (Incremental) Hashi Puzzle Solver.

This module provides functionality to solve Hashi puzzles incrementally
by predicting one bridge addition at a time.
"""
import torch
from torch_geometric.data import Data
from typing import Tuple, Optional, Dict, Any

from .train_utils import update_node_features, select_ar_action, apply_ar_action
from .utils import check_puzzle_solved


@torch.no_grad()
def solve_incremental(
    model: torch.nn.Module,
    initial_state: Data,
    max_steps: int = 30,
    head_type: str = 'regression',
    model_config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, int]:
    """
    Solve a Hashi puzzle incrementally using AR predictions.

    Args:
        model: Trained AR model
        initial_state: PyG Data object with puzzle state
        max_steps: Maximum number of steps to attempt
        head_type: 'regression' or 'conditional'
        model_config: Model configuration dictionary

    Returns:
        Tuple of (solved: bool, steps_taken: int)
    """
    if model_config is None:
        model_config = {}

    # Initialize with empty bridge state
    current_bridges = torch.zeros_like(initial_state.y)
    current_state = initial_state.clone()

    for step in range(max_steps):
        # Update node features based on current bridges
        current_state.x = update_node_features(
            current_state.x,
            current_bridges,
            current_state.edge_index,
            current_state.node_type,
            model_config
        )

        # Forward pass to get action predictions
        edge_attr = getattr(current_state, 'edge_attr', None)
        output = model(
            current_state.x,
            current_state.edge_index,
            edge_attr=edge_attr,
            node_type=current_state.node_type
        )

        # Select best valid action
        edge_idx, confidence = select_ar_action(
            output,
            current_bridges,
            current_state.edge_mask,
            head_type
        )

        # No valid actions available
        if edge_idx == -1:
            break

        # Apply the selected action
        # Get the actual action value from model output
        if head_type == 'regression':
            from .models.heads import RegressionActionHead
            action, _ = RegressionActionHead.predict_action_static(output[edge_idx:edge_idx+1])
        else:  # conditional
            from .models.heads import ConditionalActionHead
            action, _ = ConditionalActionHead.predict_action_static(output[edge_idx])

        current_bridges = apply_ar_action(current_bridges, action, edge_idx)

        # Check if puzzle is solved
        if check_puzzle_solved(current_bridges, current_state, model_config):
            return True, step + 1

    # Check final state
    if check_puzzle_solved(current_bridges, current_state, model_config):
        return True, max_steps

    return False, max_steps


@torch.no_grad()
def solve_incremental_batch(
    model: torch.nn.Module,
    batch: Data,
    max_steps: int = 30,
    head_type: str = 'regression',
    model_config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve a batch of puzzles incrementally.

    Args:
        model: Trained AR model
        batch: Batch of PyG Data objects
        max_steps: Maximum steps per puzzle
        head_type: 'regression' or 'conditional'
        model_config: Model configuration

    Returns:
        Tuple of (solved_mask: [num_graphs], steps_taken: [num_graphs])
    """
    if model_config is None:
        model_config = {}

    device = next(model.parameters()).device
    batch = batch.to(device)

    num_graphs = batch.num_graphs
    solved_mask = torch.zeros(num_graphs, dtype=torch.bool, device=device)
    steps_taken = torch.zeros(num_graphs, dtype=torch.int, device=device)

    # Initialize bridge states for each puzzle in batch
    current_bridges_batch = torch.zeros_like(batch.y)

    # Get batch indices for edges
    from .train_utils import get_edge_batch_indices
    edge_batch_indices = get_edge_batch_indices(batch)

    for step in range(max_steps):
        # Update node features for all puzzles in batch
        batch.x = update_node_features(
            batch.x,
            current_bridges_batch,
            batch.edge_index,
            batch.node_type,
            model_config
        )

        # Forward pass
        edge_attr = getattr(batch, 'edge_attr', None)
        output = model(
            batch.x,
            batch.edge_index,
            edge_attr=edge_attr,
            batch=batch.batch,
            node_type=batch.node_type
        )

        # Process each puzzle in the batch
        for graph_idx in range(num_graphs):
            if solved_mask[graph_idx]:
                continue  # Already solved

            # Get edges for this puzzle
            puzzle_edge_mask = edge_batch_indices == graph_idx
            puzzle_output = output[puzzle_edge_mask]
            puzzle_bridges = current_bridges_batch[puzzle_edge_mask]
            puzzle_edge_mask_bool = batch.edge_mask[puzzle_edge_mask]

            # Select action for this puzzle
            edge_idx_local, confidence = select_ar_action(
                puzzle_output,
                puzzle_bridges,
                puzzle_edge_mask_bool,
                head_type
            )

            if edge_idx_local == -1:
                continue  # No valid actions

            # Convert local edge index to global
            global_edge_idx = torch.where(puzzle_edge_mask)[0][edge_idx_local]

            # Get action value
            if head_type == 'regression':
                from .models.heads import RegressionActionHead
                action, _ = RegressionActionHead.predict_action_static(puzzle_output[edge_idx_local:edge_idx_local+1])
            else:  # conditional
                from .models.heads import ConditionalActionHead
                action, _ = ConditionalActionHead.predict_action_static(puzzle_output[edge_idx_local:edge_idx_local+1])

            # Apply action
            current_bridges_batch = apply_ar_action(current_bridges_batch, action, global_edge_idx)

            # Check if this puzzle is solved
            # Extract puzzle data for checking
            puzzle_edges = puzzle_edge_mask
            puzzle_data = Data(
                x=batch.x[batch.batch == graph_idx],
                edge_index=batch.edge_index[:, puzzle_edges],
                y=batch.y[puzzle_edges],
                node_type=batch.node_type[batch.batch == graph_idx],
                edge_mask=batch.edge_mask[puzzle_edges]
            )
            puzzle_bridges_final = current_bridges_batch[puzzle_edges]

            if check_puzzle_solved(puzzle_bridges_final, puzzle_data, model_config):
                solved_mask[graph_idx] = True
                steps_taken[graph_idx] = step + 1

    # Mark unsolved puzzles with max_steps
    unsolved_mask = ~solved_mask
    steps_taken[unsolved_mask] = max_steps

    return solved_mask, steps_taken
