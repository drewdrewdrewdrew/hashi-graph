"""Utilities for Auto-Regressive (AR) rollout and component management."""
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter


def get_edge_feature_indices(model_config: dict) -> dict[str, int]:
    """
    Get the indices of edge features in edge_attr.

    Args:
        model_config: Model configuration dictionary.

    Returns
    -------
    dict[str, int]
        Mapping from feature name to its index in edge_attr.
    """
    edge_map = {}
    current_idx = 0
    # Base: inv_dx, inv_dy, is_meta
    edge_map["inv_dx"] = current_idx
    edge_map["inv_dy"] = current_idx + 1
    edge_map["is_meta"] = current_idx + 2
    current_idx += 3

    if model_config.get("use_conflict_edges", False):
        edge_map["is_conflict"] = current_idx
        current_idx += 1
    if model_config.get("use_meta_mesh", False):
        edge_map["is_meta_mesh"] = current_idx
        current_idx += 1
    if model_config.get("use_meta_row_col_edges", False):
        edge_map["is_meta_row_col_cross"] = current_idx
        current_idx += 1
    if model_config.get("use_edge_labels_as_features", False):
        edge_map["bridge_label"] = current_idx
        edge_map["is_labeled"] = current_idx + 1
        current_idx += 2
    if model_config.get("use_cut_edges", False):
        edge_map["is_cut_edge"] = current_idx
        current_idx += 1

    return edge_map


def detect_components(
    num_islands: int,
    edge_index: torch.Tensor,
    current_bridges: torch.Tensor,
    node_type: torch.Tensor,
) -> torch.Tensor:
    """
    Detect connected components of islands based on current bridges.

    Args:
        num_islands: Number of islands in the graph.
        edge_index: Graph connectivity [2, num_edges].
        current_bridges: Current bridge counts [num_edges].
        node_type: Node types [num_nodes].

    Returns
    -------
    torch.Tensor
        Component representative for each island [num_islands].
    """
    # Only consider puzzle edges (both ends are islands) with bridges > 0
    row, col = edge_index
    mask = (node_type[row] <= 8) & (node_type[col] <= 8) & (current_bridges > 0)

    active_edges = edge_index[:, mask]

    if active_edges.size(1) == 0:
        return torch.arange(num_islands, device=edge_index.device)

    # Use scipy's connected_components for vectorized calculation
    # We move to CPU because scipy operates on numpy arrays
    # This is much faster than a Python DSU loop even with CPU/GPU transfer
    adj_row = active_edges[0].cpu().numpy()
    adj_col = active_edges[1].cpu().numpy()
    data = torch.ones(active_edges.size(1)).cpu().numpy()

    # Create sparse adjacency matrix
    adj = csr_matrix((data, (adj_row, adj_col)), shape=(num_islands, num_islands))

    # Calculate components
    n_components, labels = connected_components(
        csgraph=adj, directed=False, return_labels=True
    )

    labels_torch = torch.from_numpy(labels).to(edge_index.device).long()

    # Remap labels to the first island index in each component to maintain
    # the property that the representative is an island index.
    # This ensures consistency with existing tests and logic.
    island_indices = torch.arange(num_islands, device=edge_index.device)
    first_indices = scatter(
        island_indices, labels_torch, dim=0, dim_size=n_components, reduce="min"
    )
    representatives = first_indices[labels_torch]

    return representatives


def rewire_component_meta_edges(data: Data, representatives: torch.Tensor) -> Data:
    """
    Rewire island-to-component-meta edges based on current components.

    Args:
        data: PyG Data object.
        representatives: Component representative for each island [num_islands].

    Returns
    -------
    Data
        Updated Data object.
    """
    num_islands = representatives.size(0)
    node_type = data.node_type
    edge_index = data.edge_index

    # Component meta nodes were added right after islands: N to 2N-1
    # Find the indices of the component meta edges in edge_index
    row, col = edge_index

    # Component meta edges connect an island to a component meta node.
    # Initially, they connect island i to meta node N+i.
    # After components are merged, we update them to connect to the representative.

    # Mask for component meta edges: one end is island, other is component meta
    mask = (node_type[row] <= 8) & (node_type[col] == 11)

    # Update the component meta node index for each island
    # If island i is in a component represented by r, it connects to meta node N+r.
    meta_start_idx = num_islands
    islands_in_mask = row[mask]
    data.edge_index[1, mask] = meta_start_idx + representatives[islands_in_mask]

    # Also handle bidirectional edges if they exist
    rev_mask = (node_type[row] == 11) & (node_type[col] <= 8)
    data.edge_index[0, rev_mask] = (
        meta_start_idx + representatives[col[rev_mask]]
    )

    return data


def rewire_component_meta_edges_batch(
    collated_data: Batch,
    active_puzzles: list,
) -> None:
    """
    Batched wrapper that rewires component meta edges for every puzzle.

    Operates in-place on ``collated_data.edge_index``.
    """
    device = collated_data.edge_index.device

    # Node offsets provided by PyG Batch; synthesize edge offsets ourselves.
    node_offsets = collated_data.ptr[:-1]
    edge_counts = torch.tensor([p.num_edges for p in active_puzzles], device=device)
    edge_offsets = torch.cat(
        [
            torch.zeros(1, device=device, dtype=edge_counts.dtype),
            torch.cumsum(edge_counts, dim=0)[:-1],
        ]
    )

    for i, puzzle in enumerate(active_puzzles):
        start_e = edge_offsets[i]
        end_e = start_e + puzzle.num_edges

        local_edge_index = (
            collated_data.edge_index[:, start_e:end_e] - node_offsets[i]
        )
        local_bridges = puzzle.current_bridges
        local_node_type = collated_data.node_type[
            collated_data.ptr[i] : collated_data.ptr[i + 1]
        ]

        reps = detect_components(
            puzzle.num_islands,
            local_edge_index,
            local_bridges,
            local_node_type,
        )

        puzzle_edge_mask = (collated_data.edge_index[0] >= collated_data.ptr[i]) & (
            collated_data.edge_index[0] < collated_data.ptr[i + 1]
        )

        meta_start_local = puzzle.num_islands

        # Forward meta edges: [island, comp_meta]
        fwd_meta_mask = (
            (collated_data.node_type[collated_data.edge_index[0]] <= 8)
            & (collated_data.node_type[collated_data.edge_index[1]] == 11)
            & puzzle_edge_mask
        )

        islands_global = collated_data.edge_index[0, fwd_meta_mask]
        islands_local = islands_global - collated_data.ptr[i]
        collated_data.edge_index[1, fwd_meta_mask] = (
            collated_data.ptr[i] + meta_start_local + reps[islands_local]
        )

        # Backward meta edges: [comp_meta, island]
        rev_meta_mask = (
            (collated_data.node_type[collated_data.edge_index[0]] == 11)
            & (collated_data.node_type[collated_data.edge_index[1]] <= 8)
            & puzzle_edge_mask
        )

        islands_global_rev = collated_data.edge_index[1, rev_meta_mask]
        islands_local_rev = islands_global_rev - collated_data.ptr[i]
        collated_data.edge_index[0, rev_meta_mask] = (
            collated_data.ptr[i] + meta_start_local + reps[islands_local_rev]
        )
