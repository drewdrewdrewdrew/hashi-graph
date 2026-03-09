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

    # Base: inv_dx, inv_dy
    edge_map["inv_dx"] = current_idx
    edge_map["inv_dy"] = current_idx + 1
    current_idx += 2

    if not model_config.get("use_categorical_edge_types", False):
        edge_map["is_meta"] = current_idx
        current_idx += 1

        if model_config.get("use_component_meta", False):
            edge_map["is_comp_membership"] = current_idx
            edge_map["is_comp_hierarchy"] = current_idx + 1
            current_idx += 2

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
    if model_config.get("use_potential_crossing", False):
        edge_map["is_potential_crossing"] = current_idx
        current_idx += 1
    if model_config.get("use_continuous_edge_labels", False):
        edge_map["bridge_logits"] = current_idx  # Starts a 3-wide block
        current_idx += 3

    return edge_map


def detect_components(
    num_islands: int,
    edge_index: torch.Tensor,
    current_bridges: torch.Tensor,
    node_type: torch.Tensor,
    logits: torch.Tensor | None = None,
    margin: float | None = None,
) -> torch.Tensor:
    """
    Detect connected components of islands based on current bridges or logits.

    Args:
        num_islands: Number of islands in the graph.
        edge_index: Graph connectivity [2, num_edges].
        current_bridges: Current bridge counts [num_edges].
        node_type: Node types [num_nodes].
        logits: Raw edge logits [num_edges, 3] for probabilistic AM logic.
        margin: Probability margin for AM logic.

    Returns
    -------
    torch.Tensor
        Component representative for each island [num_islands].
    """
    # Only consider puzzle edges (both ends are islands)
    row, col = edge_index
    puzzle_edge_mask = (node_type[row] <= 8) & (node_type[col] <= 8)

    if logits is not None and margin is not None:
        # Probabilistic AM logic (Membership)
        # argmax != 0 AND (prob_best - prob_2nd) > margin
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        top2_probs, _ = probs.topk(2, dim=-1)
        prob_margin = top2_probs[:, 0] - top2_probs[:, 1]
        
        mask = puzzle_edge_mask & (preds > 0) & (prob_margin > margin)
    else:
        # Fallback to discrete logic
        mask = puzzle_edge_mask & (current_bridges > 0)

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
    logits: torch.Tensor | None = None,
    margin: float | None = None,
) -> None:
    """
    Batched wrapper that rewires component meta edges for every puzzle.

    Operates in-place on ``collated_data.edge_index``.
    """
    device = collated_data.edge_index.device
    edge_batch = collated_data.batch[collated_data.edge_index[0]]

    for i, puzzle in enumerate(active_puzzles):
        # Identify edges belonging to this specific puzzle in the batch
        puzzle_mask = (edge_batch == i)
        
        local_edge_index = (
            collated_data.edge_index[:, puzzle_mask] - collated_data.ptr[i]
        )
        local_bridges = puzzle.current_bridges
        local_node_type = collated_data.node_type[
            collated_data.ptr[i] : collated_data.ptr[i + 1]
        ]
        
        local_logits = None
        if logits is not None:
            local_logits = logits[puzzle_mask]

        reps = detect_components(
            puzzle.num_islands,
            local_edge_index,
            local_bridges,
            local_node_type,
            logits=local_logits,
            margin=margin,
        )

        meta_start_local = puzzle.num_islands

        # Forward meta edges: [island, comp_meta]
        # Only search within the identified puzzle edges
        fwd_meta_mask = (
            (collated_data.node_type[collated_data.edge_index[0]] <= 8)
            & (collated_data.node_type[collated_data.edge_index[1]] == 11)
            & puzzle_mask
        )

        if fwd_meta_mask.any():
            islands_global = collated_data.edge_index[0, fwd_meta_mask]
            islands_local = islands_global - collated_data.ptr[i]
            collated_data.edge_index[1, fwd_meta_mask] = (
                collated_data.ptr[i] + meta_start_local + reps[islands_local]
            )

        # Backward meta edges: [comp_meta, island]
        rev_meta_mask = (
            (collated_data.node_type[collated_data.edge_index[0]] == 11)
            & (collated_data.node_type[collated_data.edge_index[1]] <= 8)
            & puzzle_mask
        )

        if rev_meta_mask.any():
            islands_global_rev = collated_data.edge_index[1, rev_meta_mask]
            islands_local_rev = islands_global_rev - collated_data.ptr[i]
            collated_data.edge_index[0, rev_meta_mask] = (
                collated_data.ptr[i] + meta_start_local + reps[islands_local_rev]
            )


def rewire_hierarchical_edges(
    collated_data: Batch,
    model_config: dict,
    active_puzzles: list | None = None,
    current_bridges: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
) -> Batch:
    """
    Perform hierarchical rewiring of component meta nodes.

    1. Update island -> component_meta edges (standard rewiring).
    2. Add comp_meta <-> comp_meta edges for boundary puzzle edges.
    3. Add comp_meta <-> global_meta edges for active representatives.

    Returns a NEW Batch object with updated edge_index and edge_attr.
    """
    margin = model_config.get("component_merge_margin")
    device = collated_data.edge_index.device

    # 0. Clean up existing hierarchical edges from previous steps
    # Hierarchical edges are exclusively (11, 11), (11, 9), or (9, 11)
    row, col = collated_data.edge_index
    node_type = collated_data.node_type
    hier_mask = (
        ((node_type[row] == 11) & (node_type[col] == 11)) |
        ((node_type[row] == 11) & (node_type[col] == 9)) |
        ((node_type[row] == 9) & (node_type[col] == 11))
    )
    if hier_mask.any():
        keep_mask = ~hier_mask
        collated_data.edge_index = collated_data.edge_index[:, keep_mask]
        collated_data.edge_attr = collated_data.edge_attr[keep_mask]
        
        if hasattr(collated_data, "y") and collated_data.y is not None:
            collated_data.y = collated_data.y[keep_mask]
        if hasattr(collated_data, "edge_mask") and collated_data.edge_mask is not None:
            collated_data.edge_mask = collated_data.edge_mask[keep_mask]
        if hasattr(collated_data, "velocity_target") and collated_data.velocity_target is not None:
            collated_data.velocity_target = collated_data.velocity_target[keep_mask]
            
        if logits is not None:
            logits = logits[keep_mask]
        if current_bridges is not None:
            current_bridges = current_bridges[keep_mask]

    # If active_puzzles is not provided, we must synthesize it from collated_data
    # and current_bridges.
    if active_puzzles is None:
        if current_bridges is None:
            msg = "Either active_puzzles or current_bridges must be provided."
            raise ValueError(msg)

        # Synthesize active_puzzles-like objects
        active_puzzles = []
        node_ptr = collated_data.ptr
        # RE-CALCULATE edge_batch after cleanup
        edge_batch = collated_data.batch[collated_data.edge_index[0]]
        
        for i in range(collated_data.num_graphs):
            start_n, end_n = node_ptr[i], node_ptr[i+1]
            local_node_type = collated_data.node_type[start_n:end_n]
            num_islands = (local_node_type <= 8).sum().item()
            
            # Find edges belonging to this puzzle
            puzzle_edge_mask = (edge_batch == i)
            num_edges = puzzle_edge_mask.sum().item()
            puzzle_bridges = current_bridges[puzzle_edge_mask]
            
            # Simple wrapper object
            class _P:
                pass
            p = _P()
            p.num_edges = num_edges
            p.num_islands = num_islands
            p.current_bridges = puzzle_bridges
            active_puzzles.append(p)

    # 1. Standard rewiring (modifies collated_data.edge_index in-place)
    rewire_component_meta_edges_batch(
        collated_data, active_puzzles, logits=logits, margin=margin
    )

    if not model_config.get("use_hierarchical_component_meta", False):
        return collated_data

    device = collated_data.edge_index.device
    num_edge_feats = collated_data.edge_attr.size(1)
    edge_map = get_edge_feature_indices(model_config)
    is_meta_idx = edge_map.get("is_meta")

    new_edges = []
    new_attrs = []
    new_types = []

    edge_batch = collated_data.batch[collated_data.edge_index[0]]

    for i, puzzle in enumerate(active_puzzles):
        puzzle_mask = (edge_batch == i)
        start_n = collated_data.ptr[i]
        end_n = collated_data.ptr[i + 1]

        local_edge_index = (
            collated_data.edge_index[:, puzzle_mask] - start_n
        )
        local_bridges = puzzle.current_bridges
        local_node_type = collated_data.node_type[start_n:end_n]
        
        local_logits = None
        if logits is not None:
            local_logits = logits[puzzle_mask]

        # Already calculated in rewire_component_meta_edges_batch, but we need it here
        reps = detect_components(
            puzzle.num_islands,
            local_edge_index,
            local_bridges,
            local_node_type,
            logits=local_logits,
            margin=margin,
        )

        num_islands = puzzle.num_islands
        meta_start_local = num_islands

        # 2. Identify Boundary Edges: puzzle edges with label 0 connecting diff comps
        row, col = local_edge_index
        # Only consider puzzle edges (island <-> island)
        puzzle_edge_mask = (local_node_type[row] <= 8) & (local_node_type[col] <= 8)
        
        if local_logits is not None and margin is not None:
            # Probabilistic BM logic: any edge NOT AM
            # AM = (preds > 0) & (prob_margin > margin)
            # BM = ~AM
            probs = torch.softmax(local_logits, dim=-1)
            preds = local_logits.argmax(dim=-1)
            top2_probs, _ = probs.topk(2, dim=-1)
            prob_margin = top2_probs[:, 0] - top2_probs[:, 1]
            
            am_mask = (preds > 0) & (prob_margin > margin)
            boundary_mask = puzzle_edge_mask & (~am_mask)
        else:
            boundary_mask = puzzle_edge_mask & (local_bridges == 0)

        # Boundary edges connect different components
        comp_u = reps[row[boundary_mask]]
        comp_v = reps[col[boundary_mask]]
        diff_comp_mask = comp_u < comp_v # Use < to handle each pair once and ignore self-loops

        if diff_comp_mask.any():
            u_reps = comp_u[diff_comp_mask]
            v_reps = comp_v[diff_comp_mask]
            
            # Ensure unique pairs of components
            # We can use a set of frozensets or just unique on stacked reps
            pairs = torch.stack([u_reps, v_reps], dim=1)
            unique_pairs = torch.unique(pairs, dim=0)
            u_reps, v_reps = unique_pairs[:, 0], unique_pairs[:, 1]

            # Create bidirectional edges between Component Meta nodes
            # Comp Meta index is meta_start_local + rep
            meta_u = start_n + meta_start_local + u_reps
            meta_v = start_n + meta_start_local + v_reps

            new_edges.append(torch.stack([meta_u, meta_v], dim=0))
            new_edges.append(torch.stack([meta_v, meta_u], dim=0))

            # Edge attributes
            num_new = u_reps.size(0) * 2
            feat = torch.zeros((num_new, num_edge_feats), device=device)
            if is_meta_idx is not None:
                feat[:, is_meta_idx] = 1.0
            new_attrs.append(feat)
            
            # New Categorical Types
            if model_config.get("use_categorical_edge_types", False):
                new_types.append(torch.full((num_new,), 3, dtype=torch.long, device=device))

        # 3. Create Comp <-> Global Edges: active representative -> global meta
        active_reps = torch.unique(reps)
        global_meta_local = torch.where(local_node_type == 9)[0]

        if global_meta_local.numel() > 0:
            g_idx = start_n + global_meta_local[0]
            meta_reps = start_n + meta_start_local + active_reps

            # Comp -> Global
            new_edges.append(
                torch.stack(
                    [meta_reps, g_idx.expand_as(meta_reps)], dim=0
                )
            )
            # Global -> Comp
            new_edges.append(
                torch.stack(
                    [g_idx.expand_as(meta_reps), meta_reps], dim=0
                )
            )

            num_new = active_reps.size(0) * 2
            feat = torch.zeros((num_new, num_edge_feats), device=device)
            if is_meta_idx is not None:
                feat[:, is_meta_idx] = 1.0
            new_attrs.append(feat)

            # New Categorical Types
            if model_config.get("use_categorical_edge_types", False):
                new_types.append(torch.full((num_new,), 4, dtype=torch.long, device=device))

    if new_edges:
        all_new_edges = torch.cat(new_edges, dim=1)
        all_new_attrs = torch.cat(new_attrs, dim=0)
        num_new = all_new_edges.size(1)

        # Update edge_index and edge_attr
        collated_data.edge_index = torch.cat(
            [collated_data.edge_index, all_new_edges], dim=1
        )
        collated_data.edge_attr = torch.cat(
            [collated_data.edge_attr, all_new_attrs], dim=0
        )

        # Update edge_type if present
        if hasattr(collated_data, "edge_type") and collated_data.edge_type is not None:
            all_new_types = torch.cat(new_types, dim=0)
            collated_data.edge_type = torch.cat(
                [collated_data.edge_type, all_new_types], dim=0
            )

        # Update edge_mask if present (new edges are meta, so mask is False)
        if hasattr(collated_data, "edge_mask") and collated_data.edge_mask is not None:
            new_mask = torch.zeros(num_new, dtype=torch.bool, device=device)
            collated_data.edge_mask = torch.cat(
                [collated_data.edge_mask, new_mask], dim=0
            )

        # Update labels y if present (new edges have label 0)
        if hasattr(collated_data, "y") and collated_data.y is not None:
            new_y = torch.zeros(num_new, dtype=torch.long, device=device)
            collated_data.y = torch.cat([collated_data.y, new_y], dim=0)

    return collated_data