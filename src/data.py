"""
Custom PyTorch Geometric dataset for Hashi puzzle graphs.
"""
import json
from pathlib import Path
from typing import List, Optional

import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm


class MakeBidirectional(object):
    """
    Reconstructs reverse edges from a graph that only stores one direction (source < target).
    Also reconstructs symmetric edge attributes (negating dx/dy for reverse edges).
    """
    def __call__(self, data):
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            return data
            
        row, col = data.edge_index
        
        # 1. Create reverse edges
        rev_edge_index = torch.stack([col, row], dim=0)
        data.edge_index = torch.cat([data.edge_index, rev_edge_index], dim=1)
        
        # 2. Duplicate labels (y)
        if hasattr(data, 'y') and data.y is not None:
            data.y = torch.cat([data.y, data.y], dim=0)
            
        # 3. Duplicate edge_mask
        if hasattr(data, 'edge_mask') and data.edge_mask is not None:
            data.edge_mask = torch.cat([data.edge_mask, data.edge_mask], dim=0)
            
        # 4. Handle edge_attr (negate only inv_dx and inv_dy)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            fwd_attr = data.edge_attr
            rev_attr = fwd_attr.clone()
            # Negate inv_dx (idx 0) and inv_dy (idx 1)
            rev_attr[:, 0] *= -1
            rev_attr[:, 1] *= -1
            data.edge_attr = torch.cat([fwd_attr, rev_attr], dim=0)
            
        return data


class GridStretch(object):
    """
    Randomly inserts an empty gap row or column into the puzzle to vary distances.
    Updates pos and recalculates inv_dx/inv_dy features.
    """
    def __init__(self, prob=0.5, max_gap=3):
        self.prob = prob
        self.max_gap = max_gap

    def __call__(self, data):
        if torch.rand(1) > self.prob:
            return data

        # Choose axis: 0=vertical gap (stretch x), 1=horizontal gap (stretch y)
        axis = torch.randint(0, 2, (1,)).item()
        
        # Get coordinates for this axis
        # Use only valid coordinates (ignore the -1000.0 markers for meta nodes)
        valid_mask = data.pos[:, axis] > -500
        if not valid_mask.any():
            return data
            
        coords = data.pos[valid_mask, axis]
        unique_coords = torch.unique(coords)
        
        if len(unique_coords) < 2: 
            return data
        
        # Pick a split point (between two existing coords)
        # We pick an index in the sorted unique coords
        split_idx = torch.randint(0, len(unique_coords) - 1, (1,)).item()
        split_val = unique_coords[split_idx]
        gap_size = torch.randint(1, self.max_gap + 1, (1,)).item()
        
        # Shift everything on the far side of the split
        # This affects ALL nodes that have a valid coordinate > split_val
        shift_mask = (data.pos[:, axis] > -500) & (data.pos[:, axis] > split_val)
        data.pos[shift_mask, axis] += gap_size
        
        # RE-CALCULATE ALL inv_dx, inv_dy features for edges
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            row, col = data.edge_index
            diffs = data.pos[col] - data.pos[row]
            
            # Recalculate inv_dx, inv_dy
            # sign(d) / (|d| + eps)
            new_inv_dx = torch.sign(diffs[:, 0]) / (torch.abs(diffs[:, 0]) + 1e-6)
            new_inv_dy = torch.sign(diffs[:, 1]) / (torch.abs(diffs[:, 1]) + 1e-6)
            
            # Only update where the original diff indicates a relationship
            # We assume non-zero dx/dy in edge_attr implies an existing spatial relationship
            # However, since we might have stretched 0-distance (meta) edges, we should be careful.
            # BUT: In the original data, meta-edges (like global-meta) have inv_dx=0.
            # If we stretch them, diffs will be large. But logically they should stay 0?
            # Actually, Global Meta is at -1000. Distance will be huge.
            # We should ONLY update edges that are "spatial".
            # Spatial edges: Original edges, Meta Mesh edges.
            # These can be identified by:
            # - Original: is_meta=0, is_conflict=0
            # - Meta Mesh: is_meta=0 (in original code), is_meta_mesh=1
            
            # Let's check the flags in edge_attr.
            # idx 2 is is_meta. idx 3 might be is_conflict.
            # It's safer to check the current value of inv_dx/dy.
            # If it was 0, keep it 0?
            # No, if we stretch, a previously 0 distance won't become non-zero unless we moved nodes apart
            # that were at same location. But nodes at same location implies same node or meta connection.
            # If we move an island, its distance to global meta changes. But global meta edge should have dx=0.
            
            # Simple heuristic: Only update if the original inv_dx/inv_dy was non-zero?
            # No, because GridStretch increases distance, so inv_dist decreases.
            # What if we start with distance 1 (inv=1) and stretch to distance 2 (inv=0.5).
            # What about distance 0 (inv=0)?
            # If inv=0 originally, it means "no spatial relation" (e.g. meta edge).
            # So we should mask updates.
            
            mask_dx = torch.abs(data.edge_attr[:, 0]) > 1e-6
            mask_dy = torch.abs(data.edge_attr[:, 1]) > 1e-6
            
            # Update only those that were non-zero.
            # Wait! The user might want the stretch to CREATE distance features for meta edges?
            # No, the meta edges are symbolic.
            # So masking by previous non-zero value is the correct "graceful" approach.
            
            data.edge_attr[mask_dx, 0] = new_inv_dx[mask_dx]
            data.edge_attr[mask_dy, 1] = new_inv_dy[mask_dy]
            
        return data


class RandomHashiAugment(object):
    """
    Composes geometric augmentations: Rotate, Flip, Stretch.
    """
    def __init__(self, stretch_prob=0.5, max_stretch=3):
        self.stretch = GridStretch(prob=stretch_prob, max_gap=max_stretch)
        
    def __call__(self, data):
        # 1. Random Rotate 0, 90, 180, 270
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            # Rotate pos k times 90 deg counter-clockwise
            # (x, y) -> (-y, x)
            for _ in range(k):
                # Swap and negate x
                data.pos = torch.stack([-data.pos[:, 1], data.pos[:, 0]], dim=1)
                
            # If we rotated, we MUST recalculate edge features or rotate them
            # Recalculation is safer and easier
            # But we need to know which edges to update (same mask logic as Stretch)
            pass

        # 2. Random Flip
        if torch.rand(1) < 0.5:
            # Flip x
            data.pos[:, 0] *= -1
        if torch.rand(1) < 0.5:
            # Flip y
            data.pos[:, 1] *= -1
            
        # 3. Apply Stretch
        data = self.stretch(data)
        
        # 4. Final Recalculation of features (handles rotation/flip updates too)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            row, col = data.edge_index
            diffs = data.pos[col] - data.pos[row]
            
            new_inv_dx = torch.sign(diffs[:, 0]) / (torch.abs(diffs[:, 0]) + 1e-6)
            new_inv_dy = torch.sign(diffs[:, 1]) / (torch.abs(diffs[:, 1]) + 1e-6)
            
            # Apply update with mask
            mask_dx = torch.abs(data.edge_attr[:, 0]) > 1e-6
            mask_dy = torch.abs(data.edge_attr[:, 1]) > 1e-6
            
            data.edge_attr[mask_dx, 0] = new_inv_dx[mask_dx]
            data.edge_attr[mask_dy, 1] = new_inv_dy[mask_dy]

        return data


class HashiDataset(Dataset):
    """
    PyTorch Geometric dataset for Hashi puzzles.
    
    Loads puzzle graphs from a directory of JSON files.
    """
    def __init__(self, root: str, split: str = 'train',
                 size: Optional[List[int]] = None,
                 difficulty: Optional[List[int]] = None,
                 limit: Optional[int] = None,
                 use_degree: bool = False,
                 use_meta_node: bool = False,
                 use_row_col_meta: bool = False,
                 use_meta_mesh: bool = False,
                 use_meta_row_col_edges: bool = False,
                 use_distance: bool = False,
                 use_edge_labels_as_features: bool = False,
                 use_closeness_centrality: bool = False,
                 use_conflict_edges: bool = False,
                 use_capacity: bool = True,
                 use_structural_degree: bool = True,
                 use_structural_degree_nsew: bool = False,
                 use_unused_capacity: bool = True,
                 use_conflict_status: bool = True,
                 transform=None, pre_transform=None):
        """
        Args:
            root (str): Root directory where the dataset should be stored.
            split (str): The dataset split, one of 'train', 'val', or 'test'.
            size (Optional[List[int]]): List of puzzle sizes to include.
            difficulty (Optional[List[int]]): List of difficulties to include.
            limit (Optional[int]): Limit the dataset to the first `limit` files.
            use_degree (bool): Whether to include node degree as a feature. Default: False.
            use_meta_node (bool): Whether to add a meta node connected to all other nodes. Default: False.
            use_row_col_meta (bool): Whether to add row/column meta nodes. Default: False.
            use_meta_mesh (bool): Whether to connect row metas to each other and col metas to each other. Default: False.
            use_meta_row_col_edges (bool): Whether to connect each row meta to each col meta. Default: False.
            use_distance (bool): Whether to include inverse signed distance as an edge feature. Default: False.
            use_edge_labels_as_features (bool): Whether to include edge labels as input features for masking. Default: False.
            use_closeness_centrality (bool): Whether to include closeness centrality as a node feature. Default: False.
            use_conflict_edges (bool): Whether to add conflict edges for crossing constraints. Default: False.
            use_capacity (bool): Whether to include logical capacity as a node feature. Default: True.
            use_structural_degree (bool): Whether to include structural degree count (1-4) as a node feature. Default: True.
            use_structural_degree_nsew (bool): Whether to include structural degree as NSEW bitmask (0-15) as a node feature. Default: False.
            use_unused_capacity (bool): Whether to include unused capacity as a node feature. Default: True.
            use_conflict_status (bool): Whether to include conflict status as a node feature. Default: True.
            transform (callable, optional): A function/transform for the data object.
            pre_transform (callable, optional): A function/transform for the data object before saving.
        """
        self.split = split
        self.size_filter = size
        self.difficulty_filter = difficulty
        self.limit = limit
        self.use_degree = use_degree
        self.use_meta_node = use_meta_node
        self.use_row_col_meta = use_row_col_meta
        self.use_meta_mesh = use_meta_mesh
        self.use_meta_row_col_edges = use_meta_row_col_edges
        self.use_distance = use_distance
        self.use_edge_labels_as_features = use_edge_labels_as_features
        self.use_closeness_centrality = use_closeness_centrality
        self.use_conflict_edges = use_conflict_edges
        self.use_capacity = use_capacity
        self.use_structural_degree = use_structural_degree
        self.use_structural_degree_nsew = use_structural_degree_nsew
        self.use_unused_capacity = use_unused_capacity
        self.use_conflict_status = use_conflict_status

        # We must determine the raw file names before calling super().__init__()
        # so the parent class can correctly check if processing is needed.
        self._raw_filenames = self._get_filtered_filenames(root)
        
        # Instantiate bidirectional transform for use in get()
        self.make_bidirectional = MakeBidirectional()

        super().__init__(root, transform, pre_transform)

    def _get_filtered_filenames(self, root: str) -> List[str]:
        """Helper to scan and filter raw files based on instance attributes."""
        raw_dir = Path(root) / 'raw'
        filenames = []
        if not raw_dir.is_dir():
            raise RuntimeError(f"Raw data directory not found at {raw_dir}")

        for path in raw_dir.glob('puzzle_*.json'):
            with open(path, 'r') as f:
                data = json.load(f)

            if data.get('split') != self.split:
                continue
            if self.size_filter and data['generation_params'].get('size') not in self.size_filter:
                continue
            if self.difficulty_filter and data['generation_params'].get('difficulty') not in self.difficulty_filter:
                continue

            filenames.append(path.name)
        
        if not filenames:
            raise RuntimeError(f"No files found for split '{self.split}' with the given filters.")
            
        # Sort for reproducibility and apply limit
        filenames = sorted(filenames)
        if self.limit is not None:
            filenames = filenames[:int(self.limit)]

        return filenames

    @property
    def raw_file_names(self) -> List[str]:
        return self._raw_filenames

    @property
    def processed_file_names(self) -> List[str]:
        # Processed filenames are derived from raw filenames and config
        suffix = ""
        if self.use_degree:
            suffix += "_deg"
        if self.use_meta_node:
            suffix += "_meta"
        if self.use_row_col_meta:
            suffix += "_rc"
        if self.use_meta_mesh:
            suffix += "_mesh"
        if self.use_meta_row_col_edges:
            suffix += "_rcedge"
        if self.use_distance:
            suffix += "_dist"
        if self.use_edge_labels_as_features:
            suffix += "_lbl"
        if self.use_capacity:
            suffix += "_cap"
        if self.use_structural_degree:
            suffix += "_structdeg"
        if self.use_structural_degree_nsew:
            suffix += "_structdegnsew"
        if self.use_unused_capacity:
            suffix += "_unused"
        if self.use_conflict_status:
            suffix += "_conflict"

        # Add _oneway suffix to distinguish from old bidirectional files
        suffix += "_oneway"
        
        return [f"{Path(fn).stem}{suffix}.pt" for fn in self._raw_filenames]

    def len(self) -> int:
        return len(self._raw_filenames)

    def get(self, idx: int) -> Data:
        """Gets the data object at index `idx`."""
        processed_filename = self.processed_file_names[idx]
        # Set weights_only=False to allow loading Data objects.
        data = torch.load(Path(self.processed_dir) / processed_filename, weights_only=False)
        
        # Apply MakeBidirectional to reconstruct the full graph on the fly
        data = self.make_bidirectional(data)
        
        # Ensure edge_mask exists (for backward compatibility)
        if not hasattr(data, 'edge_mask') or data.edge_mask is None:
            # All edges are original if no mask exists
            data.edge_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
        
        # Ensure edge_conflicts exists (for backward compatibility)
        if not hasattr(data, 'edge_conflicts') or data.edge_conflicts is None:
            data.edge_conflicts = []
            
        return data

    def process(self):
        """Processes the raw data and saves it in the processed_dir."""
        for raw_filename, processed_filename in zip(tqdm(self.raw_file_names, desc=f"Processing {self.split} data"), self.processed_file_names):
            raw_path = Path(self.raw_dir) / raw_filename
            with open(raw_path, 'r') as f:
                puzzle_data = json.load(f)

            graph_info = puzzle_data['graph']

            # 1. Node Features - Factorized representation
            node_features = []

            # Get positions
            node_pos_list = [node['pos'] for node in graph_info['nodes']]

            # Pre-compute structural degree (potential directions based on grid position)
            # This is different from graph degree - it's the maximum possible neighbors in a grid
            structural_degrees = {}
            for node in graph_info['nodes']:
                x_pos, y_pos = node['pos']
                # Check which directions have valid positions
                north = any(n['pos'][0] == x_pos and n['pos'][1] == y_pos - 1 for n in graph_info['nodes'])
                south = any(n['pos'][0] == x_pos and n['pos'][1] == y_pos + 1 for n in graph_info['nodes'])
                west = any(n['pos'][0] == x_pos - 1 and n['pos'][1] == y_pos for n in graph_info['nodes'])
                east = any(n['pos'][0] == x_pos + 1 and n['pos'][1] == y_pos for n in graph_info['nodes'])

                if self.use_structural_degree_nsew:
                    # NSEW bitmask: Bit 0=North, Bit 1=South, Bit 2=West, Bit 3=East
                    bitmask = (north * 1) | (south * 2) | (west * 4) | (east * 8)
                    structural_degrees[node['id']] = max(1, bitmask)  # Ensure at least one bit set
                else:
                    # Legacy count-based approach
                    degree = north + south + west + east
                    structural_degrees[node['id']] = max(1, degree)  # Minimum degree of 1

            # Pre-compute conflict status (nodes involved in crossing edges)
            conflict_nodes = set()
            if 'edge_conflicts' in graph_info:
                for conflict in graph_info['edge_conflicts']:
                    # Mark all nodes involved in conflicting edges
                    conflict_nodes.add(conflict['edge1']['source'])
                    conflict_nodes.add(conflict['edge1']['target'])
                    conflict_nodes.add(conflict['edge2']['source'])
                    conflict_nodes.add(conflict['edge2']['target'])

            for node in graph_info['nodes']:
                features = []

                # Logical Capacity (1-8 for islands, 9/10 for meta)
                if self.use_capacity:
                    capacity = node['n']
                    features.append(float(capacity))

                # Structural Degree (1-4 count or 0-15 NSEW bitmask)
                if self.use_structural_degree or self.use_structural_degree_nsew:
                    struct_degree = structural_degrees[node['id']]
                    features.append(float(struct_degree))

                # Unused Capacity (starts as capacity, updated during training)
                if self.use_unused_capacity:
                    capacity = node['n']
                    unused_capacity = capacity if capacity <= 8 else 0  # Only puzzle nodes have capacity
                    features.append(float(unused_capacity))

                # Conflict Status (0-1 binary flag)
                if self.use_conflict_status:
                    conflict_status = 1.0 if node['id'] in conflict_nodes else 0.0
                    features.append(float(conflict_status))

                # Closeness centrality (continuous)
                if self.use_closeness_centrality:
                    closeness = node.get('closeness_centrality', 0.0)
                    features.append(float(closeness))

                node_features.append(features)

            x = torch.tensor(node_features, dtype=torch.float)

            source_nodes = [edge['source'] for edge in graph_info['edges']]
            target_nodes = [edge['target'] for edge in graph_info['edges']]
            edge_labels = [edge['label'] for edge in graph_info['edges']]
            
            # Map node ID to index
            node_id_to_idx = {node['id']: i for i, node in enumerate(graph_info['nodes'])}
            source_indices = [node_id_to_idx[uid] for uid in source_nodes]
            target_indices = [node_id_to_idx[uid] for uid in target_nodes]
            
            # ONEWAY STORAGE: Only store source < target
            # Note: The JSON usually has consistent ordering, but we enforce it here.
            # We must swap source/target if source > target
            final_src = []
            final_dst = []
            final_labels = []
            
            for s, t, l in zip(source_indices, target_indices, edge_labels):
                if s < t:
                    final_src.append(s)
                    final_dst.append(t)
                else:
                    final_src.append(t)
                    final_dst.append(s)
                final_labels.append(l)

            edge_index = torch.tensor([final_src, final_dst], dtype=torch.long)
            y = torch.tensor(final_labels, dtype=torch.long)
            
            # 2. Edge Attributes
            # Prepare edge attribute lists
            edge_attrs = []
            
            # Feature dimensions (in order):
            # - inv_dx, inv_dy, is_meta (always present)
            # - is_conflict (if use_conflict_edges)
            # - is_meta_mesh (if use_meta_mesh or use_meta_row_col_edges)
            # - is_meta_row_col_cross (if use_meta_mesh or use_meta_row_col_edges)
            # - bridge_label, is_labeled (if use_edge_labels_as_features)
            
            for src_idx, dst_idx, label in zip(final_src, final_dst, final_labels):
                x1, y1 = node_pos_list[src_idx]
                x2, y2 = node_pos_list[dst_idx]
                
                dx = x2 - x1
                dy = y2 - y1
                
                # Inverse Signed Distance
                inv_dx = 0.0
                if abs(dx) > 1e-6:
                    inv_dx = (1.0 if dx > 0 else -1.0) / (abs(dx) + 1e-6)
                    
                inv_dy = 0.0
                if abs(dy) > 1e-6:
                    inv_dy = (1.0 if dy > 0 else -1.0) / (abs(dy) + 1e-6)
                
                features = [inv_dx, inv_dy, 0.0]  # [inv_dx, inv_dy, is_meta]
                
                if self.use_conflict_edges:
                    features.append(0.0)
                if self.use_meta_mesh:
                    features.append(0.0)
                if self.use_meta_row_col_edges:
                    features.append(0.0)
                if self.use_edge_labels_as_features:
                    features.extend([float(label), 1.0])
                
                edge_attrs.append(features)
            
            num_original_edges = len(final_src)
            edge_mask = torch.ones(num_original_edges, dtype=torch.bool)
            
            num_nodes = len(graph_info['nodes'])

            # 2.5. Conflict Edges
            if self.use_conflict_edges and 'edge_conflicts' in graph_info:
                conflicts = graph_info['edge_conflicts']
                conflict_src = []
                conflict_dst = []
                
                for conflict in conflicts:
                    e1_src = conflict['edge1']['source']
                    e1_tgt = conflict['edge1']['target']
                    e2_src = conflict['edge2']['source']
                    e2_tgt = conflict['edge2']['target']
                    
                    # Store conflicts for all combinations
                    # We store only one-way for each pair
                    for n1 in [e1_src, e1_tgt]:
                        for n2 in [e2_src, e2_tgt]:
                            # Map to indices
                            idx1 = node_id_to_idx[n1]
                            idx2 = node_id_to_idx[n2]
                            if idx1 < idx2:
                                conflict_src.append(idx1)
                                conflict_dst.append(idx2)
                            else:
                                conflict_src.append(idx2)
                                conflict_dst.append(idx1)
                
                if conflict_src:
                    conflict_edge_index = torch.tensor([conflict_src, conflict_dst], dtype=torch.long)
                    edge_index = torch.cat([edge_index, conflict_edge_index], dim=1)
                    
                    num_conflict = len(conflict_src)
                    for _ in range(num_conflict):
                        features = [0.0, 0.0, 0.0, 1.0]  # [..., is_meta=0, is_conflict=1]
                        if self.use_meta_mesh: features.append(0.0)
                        if self.use_meta_row_col_edges: features.append(0.0)
                        if self.use_edge_labels_as_features: features.extend([0.0, 0.0])
                        edge_attrs.append(features)
                    
                    edge_mask = torch.cat([edge_mask, torch.zeros(num_conflict, dtype=torch.bool)])

            # 3. Global Meta Node
            if self.use_meta_node:
                num_puzzle_nodes = len(graph_info['nodes'])

                # Factorized features for global meta node (respecting enabled features)
                meta_features = []
                if self.use_capacity:
                    meta_features.append(9.0)  # Sentinel for global meta
                if self.use_structural_degree or self.use_structural_degree_nsew:
                    meta_features.append(0.0)  # Meta nodes don't have structural degree
                if self.use_unused_capacity:
                    meta_features.append(0.0)  # Meta nodes don't have capacity
                if self.use_conflict_status:
                    meta_features.append(0.0)  # Meta nodes not in conflicts
                if self.use_closeness_centrality:
                    meta_features.append(0.0)  # No closeness for meta nodes

                meta_feat_tensor = torch.tensor([meta_features], dtype=torch.float)
                
                x = torch.cat([x, meta_feat_tensor], dim=0)
                
                # Meta Node Position: Dummy values
                node_pos_list.append([-1000.0, -1000.0])
                
                meta_idx = num_nodes
                num_nodes += 1
                
                # Edges: Node -> Meta (One way)
                orig_indices = list(range(num_puzzle_nodes))
                meta_src = orig_indices
                meta_dst = [meta_idx] * len(orig_indices)
                
                meta_edge_index = torch.tensor([meta_src, meta_dst], dtype=torch.long)
                edge_index = torch.cat([edge_index, meta_edge_index], dim=1)
                
                num_meta = len(meta_src)
                for _ in range(num_meta):
                    features = [0.0, 0.0, 1.0]  # [..., is_meta=1]
                    if self.use_conflict_edges: features.append(0.0)
                    if self.use_meta_mesh: features.append(0.0)
                    if self.use_meta_row_col_edges: features.append(0.0)
                    if self.use_edge_labels_as_features: features.extend([0.0, 0.0])
                    edge_attrs.append(features)
                
                edge_mask = torch.cat([edge_mask, torch.zeros(len(meta_src), dtype=torch.bool)])

            # 4. Row/Col Meta Nodes
            if self.use_row_col_meta:
                rows = sorted(list(set(n['pos'][1] for n in graph_info['nodes'])))
                cols = sorted(list(set(n['pos'][0] for n in graph_info['nodes'])))
                
                row_map = {r: i + num_nodes for i, r in enumerate(rows)}
                
                # Add Row Meta Positions
                for r in rows:
                    node_pos_list.append([-1000.0, float(r)])
                
                num_nodes += len(rows)
                col_map = {c: i + num_nodes for i, c in enumerate(cols)}
                
                # Add Col Meta Positions
                for c in cols:
                    node_pos_list.append([float(c), -1000.0])
                
                num_nodes += len(cols)
                
                row_counts = {r: 0 for r in rows}
                col_counts = {c: 0 for c in cols}
                for node in graph_info['nodes']:
                    row_counts[node['pos'][1]] += 1
                    col_counts[node['pos'][0]] += 1
                
                # Add row meta nodes features (factorized, respecting enabled features)
                row_feats = []
                for r in rows:
                    row_features = []
                    if self.use_capacity:
                        row_features.append(10.0)  # Sentinel for row meta
                    if self.use_structural_degree or self.use_structural_degree_nsew:
                        row_features.append(0.0)  # Meta nodes don't have structural degree
                    if self.use_unused_capacity:
                        row_features.append(0.0)  # Meta nodes don't have capacity
                    if self.use_conflict_status:
                        row_features.append(0.0)  # Meta nodes not in conflicts
                    if self.use_closeness_centrality:
                        row_features.append(0.0)  # No closeness for meta nodes

                    row_feats.append(row_features)
                row_feats = torch.tensor(row_feats, dtype=torch.float)

                x = torch.cat([x, row_feats], dim=0)

                # Add col meta nodes features (factorized, respecting enabled features)
                col_feats = []
                for c in cols:
                    col_features = []
                    if self.use_capacity:
                        col_features.append(10.0)  # Sentinel for col meta
                    if self.use_structural_degree or self.use_structural_degree_nsew:
                        col_features.append(0.0)  # Meta nodes don't have structural degree
                    if self.use_unused_capacity:
                        col_features.append(0.0)  # Meta nodes don't have capacity
                    if self.use_conflict_status:
                        col_features.append(0.0)  # Meta nodes not in conflicts
                    if self.use_closeness_centrality:
                        col_features.append(0.0)  # No closeness for meta nodes

                    col_feats.append(col_features)
                col_feats = torch.tensor(col_feats, dtype=torch.float)
                
                x = torch.cat([x, col_feats], dim=0)
                
                # RC Meta Edges: Node -> Meta (One way)
                rc_src = []
                rc_dst = []
                
                for i, node in enumerate(graph_info['nodes']):
                    # Connect to row meta node
                    r_meta = row_map[node['pos'][1]]
                    # i < r_meta usually (nodes came first)
                    rc_src.append(i)
                    rc_dst.append(r_meta)
                    
                    # Connect to col meta node
                    c_meta = col_map[node['pos'][0]]
                    rc_src.append(i)
                    rc_dst.append(c_meta)
                
                rc_edge_index = torch.tensor([rc_src, rc_dst], dtype=torch.long)
                edge_index = torch.cat([edge_index, rc_edge_index], dim=1)
                
                num_rc = len(rc_src)
                for _ in range(num_rc):
                    features = [0.0, 0.0, 1.0]  # [..., is_meta=1]
                    if self.use_conflict_edges: features.append(0.0)
                    if self.use_meta_mesh: features.append(0.0)
                    if self.use_meta_row_col_edges: features.append(0.0)
                    if self.use_edge_labels_as_features: features.extend([0.0, 0.0])
                    edge_attrs.append(features)
                
                edge_mask = torch.cat([edge_mask, torch.zeros(len(rc_src), dtype=torch.bool)])
                
                # 4a. Meta Mesh: Connect row metas to each other, col metas to each other
                if self.use_meta_mesh:
                    mesh_src = []
                    mesh_dst = []
                    mesh_distances = []
                    
                    # Connect adjacent row meta nodes (Forward only)
                    for i in range(len(rows) - 1):
                        row_i = rows[i]
                        row_j = rows[i + 1]
                        idx_i = row_map[row_i]
                        idx_j = row_map[row_j]
                        dy = row_j - row_i
                        
                        # idx_i < idx_j because we processed rows in sorted order
                        mesh_src.append(idx_i)
                        mesh_dst.append(idx_j)
                        mesh_distances.append(dy)
                    
                    # Connect adjacent col meta nodes (Forward only)
                    for i in range(len(cols) - 1):
                        col_i = cols[i]
                        col_j = cols[i + 1]
                        idx_i = col_map[col_i]
                        idx_j = col_map[col_j]
                        dx = col_j - col_i
                        
                        mesh_src.append(idx_i)
                        mesh_dst.append(idx_j)
                        mesh_distances.append(dx)
                    
                    if mesh_src:
                        mesh_edge_index = torch.tensor([mesh_src, mesh_dst], dtype=torch.long)
                        edge_index = torch.cat([edge_index, mesh_edge_index], dim=1)
                        
                        num_mesh = len(mesh_src)
                        for i in range(num_mesh):
                            d = mesh_distances[i]
                            # Row connections have dy (idx < len(rows)-1), Col connections have dx
                            if i < (len(rows) - 1):  # Row mesh
                                inv_dx = 0.0
                                inv_dy = (1.0 if d > 0 else -1.0) / (abs(d) + 1e-6)
                            else:  # Col mesh
                                inv_dx = (1.0 if d > 0 else -1.0) / (abs(d) + 1e-6)
                                inv_dy = 0.0
                            
                            features = [inv_dx, inv_dy, 0.0]  # [inv_dx, inv_dy, is_meta=0]
                            if self.use_conflict_edges: features.append(0.0)
                            if self.use_meta_mesh: features.append(1.0)  # is_meta_mesh=1
                            if self.use_meta_row_col_edges: features.append(0.0)
                            if self.use_edge_labels_as_features: features.extend([0.0, 0.0])
                            edge_attrs.append(features)
                        
                        edge_mask = torch.cat([edge_mask, torch.zeros(num_mesh, dtype=torch.bool)])
                
                # 4b. Row-Col Cross Edges
                if self.use_meta_row_col_edges:
                    cross_src = []
                    cross_dst = []
                    
                    # One way: RowMeta vs ColMeta. Order depends on construction.
                    # Rows added first, then Cols. So RowIdx < ColIdx.
                    for r in rows:
                        for c in cols:
                            row_idx = row_map[r]
                            col_idx = col_map[c]
                            cross_src.append(row_idx)
                            cross_dst.append(col_idx)
                    
                    if cross_src:
                        cross_edge_index = torch.tensor([cross_src, cross_dst], dtype=torch.long)
                        edge_index = torch.cat([edge_index, cross_edge_index], dim=1)
                        
                        num_cross = len(cross_src)
                        for _ in range(num_cross):
                            features = [0.0, 0.0, 0.0]
                            if self.use_conflict_edges: features.append(0.0)
                            if self.use_meta_mesh: features.append(0.0)
                            if self.use_meta_row_col_edges: features.append(1.0) # is_meta_row_col_cross=1
                            if self.use_edge_labels_as_features: features.extend([0.0, 0.0])
                            edge_attrs.append(features)
                        
                        edge_mask = torch.cat([edge_mask, torch.zeros(num_cross, dtype=torch.bool)])
                
                # 4c. Global Meta ↔ Row/Col Meta Edges
                if self.use_meta_node:
                    global_rc_src = []
                    global_rc_dst = []
                    
                    all_line_meta_indices = list(row_map.values()) + list(col_map.values())
                    
                    # Global Meta (meta_idx) was added BEFORE Row/Col Metas?
                    # No.
                    # Order:
                    # 1. Puzzle Nodes
                    # 2. Global Meta (meta_idx)
                    # 3. Row Metas
                    # 4. Col Metas
                    # So meta_idx < line_meta_idx
                    
                    for line_meta_idx in all_line_meta_indices:
                        global_rc_src.append(meta_idx)
                        global_rc_dst.append(line_meta_idx)
                    
                    if global_rc_src:
                        global_rc_edge_index = torch.tensor([global_rc_src, global_rc_dst], dtype=torch.long)
                        edge_index = torch.cat([edge_index, global_rc_edge_index], dim=1)
                        
                        num_global_rc = len(global_rc_src)
                        for _ in range(num_global_rc):
                            features = [0.0, 0.0, 1.0]  # [..., is_meta=1]
                            if self.use_conflict_edges: features.append(0.0)
                            if self.use_meta_mesh: features.append(0.0)
                            if self.use_meta_row_col_edges: features.append(0.0)
                            if self.use_edge_labels_as_features: features.extend([0.0, 0.0])
                            edge_attrs.append(features)
                        
                        edge_mask = torch.cat([edge_mask, torch.zeros(num_global_rc, dtype=torch.bool)])
            
            # Construct final tensors
            if edge_attrs:
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            else:
                edge_attr = None # Or zeros if dimensions known

            # 5. Extract edge conflict indices (One-way indices)
            edge_conflict_indices = []
            if 'edge_conflicts' in graph_info:
                # Map (u, v) with u < v to edge index
                # We need to construct map based on edge_index we just built
                # edge_index contains: Original, Conflict, Meta, RC, Mesh, Cross, GlobalRC
                # We only care about Original edges for conflicts in Hashi usually
                # (Conflicts are between potential bridges).
                
                # Original edges are at the start.
                edge_map = {}
                for idx in range(num_original_edges):
                    s = edge_index[0, idx].item()
                    t = edge_index[1, idx].item()
                    # We know s < t from construction
                    edge_map[(s, t)] = idx
                
                for conflict in graph_info['edge_conflicts']:
                    e1_src = conflict['edge1']['source']
                    e1_tgt = conflict['edge1']['target']
                    e2_src = conflict['edge2']['source']
                    e2_tgt = conflict['edge2']['target']
                    
                    # Get IDs
                    id1_s, id1_t = node_id_to_idx[e1_src], node_id_to_idx[e1_tgt]
                    id2_s, id2_t = node_id_to_idx[e2_src], node_id_to_idx[e2_tgt]
                    
                    # Sort pairs
                    k1 = tuple(sorted((id1_s, id1_t)))
                    k2 = tuple(sorted((id2_s, id2_t)))
                    
                    e1_idx = edge_map.get(k1)
                    e2_idx = edge_map.get(k2)
                    
                    if e1_idx is not None and e2_idx is not None:
                        # Store just the pair (order doesn't matter for sets, but matters for tensor)
                        # We can store both directions if loss expects it, or just one.
                        # Existing code stored both. Let's store both to be safe for loss func.
                        edge_conflict_indices.append((e1_idx, e2_idx))
                        edge_conflict_indices.append((e2_idx, e1_idx))

            # Store node positions
            pos_tensor = torch.tensor(node_pos_list, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                       edge_mask=edge_mask, edge_conflicts=edge_conflict_indices, pos=pos_tensor)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, Path(self.processed_dir) / processed_filename)
