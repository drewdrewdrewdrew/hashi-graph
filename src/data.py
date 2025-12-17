"""
Custom PyTorch Geometric dataset for Hashi puzzle graphs.
"""
import json
from pathlib import Path
from typing import List, Optional

import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm


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

        # We must determine the raw file names before calling super().__init__()
        # so the parent class can correctly check if processing is needed.
        self._raw_filenames = self._get_filtered_filenames(root)

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
        return [f"{Path(fn).stem}{suffix}.pt" for fn in self._raw_filenames]

    def len(self) -> int:
        return len(self._raw_filenames)

    def get(self, idx: int) -> Data:
        """Gets the data object at index `idx`."""
        processed_filename = self.processed_file_names[idx]
        # Set weights_only=False to allow loading Data objects.
        # This is required for recent PyTorch versions (>=2.6) which default to True.
        data = torch.load(Path(self.processed_dir) / processed_filename, weights_only=False)
        
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

            # 1. Node Features
            node_features = []
            closeness_features = []
            
            for node in graph_info['nodes']:
                features = [node['n']]
                
                # Degree feature: Use INVERSE degree (1/degree) to bound values
                if self.use_degree:
                    if 'degree' in node:
                        degree = node['degree']
                    else:
                        # Fallback: calculate degree from edges
                        node_id = node['id']
                        degree = sum(1 for edge in graph_info['edges'] 
                                   if edge['source'] == node_id or edge['target'] == node_id)
                    # Convert to inverse degree: 1/degree (bounded in (0, 1])
                    inv_degree = 1.0 / degree if degree > 0 else 0.0
                    # Scale to ~0-100 range for embedding (like other integer features)
                    features.append(int(inv_degree * 100))
                
                node_features.append(features)
                
                # Closeness centrality: Keep as continuous float (no scaling)
                if self.use_closeness_centrality:
                    closeness_features.append(node.get('closeness_centrality', 0.0))
            
            x = torch.tensor(node_features, dtype=torch.long)
            
            # Add closeness as separate continuous feature if enabled
            if self.use_closeness_centrality:
                closeness = torch.tensor(closeness_features, dtype=torch.float).unsqueeze(1)
                # Concatenate: [embedded_features, continuous_closeness]
                x = torch.cat([x.float(), closeness], dim=1)

            source_nodes = [edge['source'] for edge in graph_info['edges']]
            target_nodes = [edge['target'] for edge in graph_info['edges']]
            edge_labels = [edge['label'] for edge in graph_info['edges']]
            
            # Map node ID to index (assuming 0 to N-1 for now, but being safe)
            # Actually, let's create a map to be safe
            node_id_to_idx = {node['id']: i for i, node in enumerate(graph_info['nodes'])}
            source_indices = [node_id_to_idx[uid] for uid in source_nodes]
            target_indices = [node_id_to_idx[uid] for uid in target_nodes]
            
            # Make original edges bidirectional
            edge_index = torch.tensor([source_indices + target_indices, target_indices + source_indices], dtype=torch.long)
            y = torch.tensor(edge_labels + edge_labels, dtype=torch.long) # Duplicate labels for reverse edges
            
            # 2. Edge Attributes (Signed Inverse Distance + Meta Flags + Conflict Flag + Optional Labels)
            edge_attrs_forward = []
            edge_attrs_reverse = []
            node_pos = {node['id']: node['pos'] for node in graph_info['nodes']}
            
            # Feature dimensions (in order):
            # - inv_dx, inv_dy, is_meta (always present)
            # - is_conflict (if use_conflict_edges)
            # - is_meta_mesh (if use_meta_mesh or use_meta_row_col_edges)
            # - is_meta_row_col_cross (if use_meta_mesh or use_meta_row_col_edges)
            # - bridge_label, is_labeled (if use_edge_labels_as_features)
            
            for src_id, dst_id, label in zip(source_nodes, target_nodes, edge_labels):
                x1, y1 = node_pos[src_id]
                x2, y2 = node_pos[dst_id]
                
                dx = x2 - x1
                dy = y2 - y1
                
                # Inverse Signed Distance: sign(d) / (|d| + epsilon)
                # Maps large distances to near 0, small distances to +/- 1
                inv_dx = 0.0
                if abs(dx) > 1e-6:
                    inv_dx = (1.0 if dx > 0 else -1.0) / (abs(dx) + 1e-6)
                    
                inv_dy = 0.0
                if abs(dy) > 1e-6:
                    inv_dy = (1.0 if dy > 0 else -1.0) / (abs(dy) + 1e-6)
                
                # Build feature vector for forward edge (src → dst)
                features_fwd = [inv_dx, inv_dy, 0.0]  # [inv_dx, inv_dy, is_meta]
                # Build feature vector for reverse edge (dst → src) with negated distances
                features_rev = [-inv_dx, -inv_dy, 0.0]
                
                # Add conflict flag if conflict edges are enabled
                if self.use_conflict_edges:
                    features_fwd.append(0.0)  # is_conflict=0 for original edges
                    features_rev.append(0.0)
                
                # Add meta mesh/row-col cross flags if enabled
                if self.use_meta_mesh:
                    features_fwd.append(0.0)  # is_meta_mesh=0 for original edges
                    features_rev.append(0.0)
                if self.use_meta_row_col_edges:
                    features_fwd.append(0.0)  # is_meta_row_col_cross=0 for original edges
                    features_rev.append(0.0)
                
                # Add labels if enabled (same label for both directions)
                if self.use_edge_labels_as_features:
                    features_fwd.extend([float(label), 1.0])  # [bridge_label, is_labeled]
                    features_rev.extend([float(label), 1.0])
                
                edge_attrs_forward.append(features_fwd)
                edge_attrs_reverse.append(features_rev)
            
            if edge_attrs_forward:
                edge_attr = torch.tensor(edge_attrs_forward + edge_attrs_reverse, dtype=torch.float)
            else:
                # Determine edge dimension based on enabled features
                edge_dim = 3  # base: [inv_dx, inv_dy, is_meta]
                if self.use_conflict_edges:
                    edge_dim += 1  # add is_conflict
                if self.use_meta_mesh:
                    edge_dim += 1  # add is_meta_mesh
                if self.use_meta_row_col_edges:
                    edge_dim += 1  # add is_meta_row_col_cross
                if self.use_edge_labels_as_features:
                    edge_dim += 2  # add bridge_label, is_labeled
                edge_attr = torch.zeros((len(source_nodes), edge_dim), dtype=torch.float)
            
            # Track original edges
            num_original_edges = len(edge_labels) * 2 # Multiply by 2 for bidirectional
            edge_mask = torch.ones(num_original_edges, dtype=torch.bool)
            
            num_nodes = len(graph_info['nodes'])

            # 2.5. Conflict Edges (before meta nodes)
            if self.use_conflict_edges and 'edge_conflicts' in graph_info:
                conflicts = graph_info['edge_conflicts']
                conflict_src = []
                conflict_dst = []
                
                for conflict in conflicts:
                    e1_src = conflict['edge1']['source']
                    e1_tgt = conflict['edge1']['target']
                    e2_src = conflict['edge2']['source']
                    e2_tgt = conflict['edge2']['target']
                    
                    # Create conflict edges between all combinations (bidirectional)
                    # If edge A-B crosses edge C-D, connect: A↔C, A↔D, B↔C, B↔D
                    for n1 in [e1_src, e1_tgt]:
                        for n2 in [e2_src, e2_tgt]:
                            conflict_src.extend([n1, n2])
                            conflict_dst.extend([n2, n1])
                
                if conflict_src:
                    conflict_edge_index = torch.tensor([conflict_src, conflict_dst], dtype=torch.long)
                    edge_index = torch.cat([edge_index, conflict_edge_index], dim=1)
                    
                    # Conflict edge attributes: is_meta=0, is_conflict=1
                    num_conflict = len(conflict_src)
                    conflict_attr_list = []
                    for _ in range(num_conflict):
                        features = [0.0, 0.0, 0.0, 1.0]  # [inv_dx, inv_dy, is_meta=0, is_conflict=1]
                        if self.use_meta_mesh:
                            features.append(0.0)  # [is_meta_mesh=0]
                        if self.use_meta_row_col_edges:
                            features.append(0.0)  # [is_meta_row_col_cross=0]
                        if self.use_edge_labels_as_features:
                            features.extend([0.0, 0.0])  # [bridge_label=0, is_labeled=0]
                        conflict_attr_list.append(features)
                    
                    conflict_attr = torch.tensor(conflict_attr_list, dtype=torch.float)
                    edge_attr = torch.cat([edge_attr, conflict_attr], dim=0)
                    
                    # CRITICAL: Conflict edges are NOT prediction targets
                    # Set edge_mask=False so they're excluded from loss computation
                    # They exist only for message passing (like meta edges)
                    edge_mask = torch.cat([edge_mask, torch.zeros(num_conflict, dtype=torch.bool)])

            # 3. Global Meta Node
            if self.use_meta_node:
                # Count puzzle nodes for degree calculation
                num_puzzle_nodes = len(graph_info['nodes'])
                
                # Meta node features: n=9 (global meta type), degree (if enabled), closeness (if enabled)
                meta_feat = [9]  # 9 = global meta node type
                if self.use_degree:
                    # Meta node degree = number of puzzle nodes it connects to (not counting row/col metas)
                    meta_degree = num_puzzle_nodes
                    meta_feat.append(int((1.0 / meta_degree) * 100))
                meta_feat_tensor = torch.tensor([meta_feat], dtype=torch.long)
                
                # Add closeness if enabled (meta node gets 0.0 closeness)
                if self.use_closeness_centrality:
                    meta_closeness = torch.tensor([[0.0]], dtype=torch.float)
                    meta_feat_tensor = torch.cat([meta_feat_tensor.float(), meta_closeness], dim=1)
                
                x = torch.cat([x, meta_feat_tensor], dim=0)
                
                meta_idx = num_nodes
                num_nodes += 1
                
                # Edges to all original nodes
                orig_indices = list(range(num_puzzle_nodes))
                meta_src = [meta_idx] * len(orig_indices) + orig_indices
                meta_dst = orig_indices + [meta_idx] * len(orig_indices)
                
                meta_edge_index = torch.tensor([meta_src, meta_dst], dtype=torch.long)
                edge_index = torch.cat([edge_index, meta_edge_index], dim=1)
                
                # Meta edge attributes: is_meta=1, is_conflict=0
                num_meta = len(meta_src)
                meta_attr_list = []
                for _ in range(num_meta):
                    features = [0.0, 0.0, 1.0]  # [inv_dx, inv_dy, is_meta=1]
                    if self.use_conflict_edges:
                        features.append(0.0)  # is_conflict=0
                    if self.use_meta_mesh:
                        features.append(0.0)  # is_meta_mesh=0
                    if self.use_meta_row_col_edges:
                        features.append(0.0)  # is_meta_row_col_cross=0
                    if self.use_edge_labels_as_features:
                        features.extend([0.0, 0.0])  # [bridge_label=0, is_labeled=0]
                    meta_attr_list.append(features)
                
                meta_edge_attr = torch.tensor(meta_attr_list, dtype=torch.float)
                edge_attr = torch.cat([edge_attr, meta_edge_attr], dim=0)
                
                # Update mask
                edge_mask = torch.cat([edge_mask, torch.zeros(len(meta_src), dtype=torch.bool)])

            # 4. Row/Col Meta Nodes
            if self.use_row_col_meta:
                # Identify unique rows and cols
                rows = sorted(list(set(n['pos'][1] for n in graph_info['nodes'])))
                cols = sorted(list(set(n['pos'][0] for n in graph_info['nodes'])))
                
                row_map = {r: i + num_nodes for i, r in enumerate(rows)}
                num_nodes += len(rows)
                col_map = {c: i + num_nodes for i, c in enumerate(cols)}
                num_nodes += len(cols)
                
                # Count puzzle nodes per row/col for accurate degree calculation
                row_counts = {r: 0 for r in rows}
                col_counts = {c: 0 for c in cols}
                for node in graph_info['nodes']:
                    row_counts[node['pos'][1]] += 1
                    col_counts[node['pos'][0]] += 1
                
                # Add row meta nodes features
                # n=10 (line meta type), degree (if enabled), closeness (if enabled)
                row_feats = []
                for r in rows:
                    row_feat = [10]  # 10 = line meta node type (row or col)
                    if self.use_degree:
                        # Row meta degree = number of puzzle nodes in that row
                        row_degree = row_counts[r]
                        row_feat.append(int((1.0 / row_degree) * 100))
                    row_feats.append(row_feat)
                row_feats = torch.tensor(row_feats, dtype=torch.long)
                
                # Add closeness if enabled
                if self.use_closeness_centrality:
                    row_closeness = torch.zeros((len(rows), 1), dtype=torch.float)
                    row_feats = torch.cat([row_feats.float(), row_closeness], dim=1)
                
                x = torch.cat([x, row_feats], dim=0)
                
                # Add col meta nodes features
                # n=10 (line meta type), degree (if enabled), closeness (if enabled)
                col_feats = []
                for c in cols:
                    col_feat = [10]  # 10 = line meta node type (row or col)
                    if self.use_degree:
                        # Col meta degree = number of puzzle nodes in that column
                        col_degree = col_counts[c]
                        col_feat.append(int((1.0 / col_degree) * 100))
                    col_feats.append(col_feat)
                col_feats = torch.tensor(col_feats, dtype=torch.long)
                
                # Add closeness if enabled
                if self.use_closeness_centrality:
                    col_closeness = torch.zeros((len(cols), 1), dtype=torch.float)
                    col_feats = torch.cat([col_feats.float(), col_closeness], dim=1)
                
                x = torch.cat([x, col_feats], dim=0)
                
                # Add edges between puzzle nodes and row/col meta nodes
                rc_src = []
                rc_dst = []
                
                for i, node in enumerate(graph_info['nodes']):
                    # Connect to row meta node
                    r_meta = row_map[node['pos'][1]]
                    rc_src.extend([i, r_meta])
                    rc_dst.extend([r_meta, i])
                    
                    # Connect to col meta node
                    c_meta = col_map[node['pos'][0]]
                    rc_src.extend([i, c_meta])
                    rc_dst.extend([c_meta, i])
                
                rc_edge_index = torch.tensor([rc_src, rc_dst], dtype=torch.long)
                edge_index = torch.cat([edge_index, rc_edge_index], dim=1)
                
                # Edge attributes for RC meta edges: is_meta=1, is_conflict=0
                num_rc = len(rc_src)
                rc_attr_list = []
                for _ in range(num_rc):
                    features = [0.0, 0.0, 1.0]  # [inv_dx, inv_dy, is_meta=1]
                    if self.use_conflict_edges:
                        features.append(0.0)  # is_conflict=0
                    if self.use_meta_mesh:
                        features.append(0.0)  # is_meta_mesh=0
                    if self.use_meta_row_col_edges:
                        features.append(0.0)  # is_meta_row_col_cross=0
                    if self.use_edge_labels_as_features:
                        features.extend([0.0, 0.0])  # [bridge_label=0, is_labeled=0]
                    rc_attr_list.append(features)
                
                rc_edge_attr = torch.tensor(rc_attr_list, dtype=torch.float)
                edge_attr = torch.cat([edge_attr, rc_edge_attr], dim=0)
                
                # Update mask
                edge_mask = torch.cat([edge_mask, torch.zeros(len(rc_src), dtype=torch.bool)])
                
                # 4a. Meta Mesh: Connect row metas to each other, col metas to each other
                if self.use_meta_mesh:
                    mesh_src = []
                    mesh_dst = []
                    mesh_distances = []
                    
                    # Connect adjacent row meta nodes (bidirectional)
                    for i in range(len(rows) - 1):
                        row_i = rows[i]
                        row_j = rows[i + 1]
                        idx_i = row_map[row_i]
                        idx_j = row_map[row_j]
                        dy = row_j - row_i  # Distance in grid coordinates
                        
                        # Bidirectional edges
                        mesh_src.extend([idx_i, idx_j])
                        mesh_dst.extend([idx_j, idx_i])
                        mesh_distances.extend([dy, -dy])
                    
                    # Connect adjacent col meta nodes (bidirectional)
                    for i in range(len(cols) - 1):
                        col_i = cols[i]
                        col_j = cols[i + 1]
                        idx_i = col_map[col_i]
                        idx_j = col_map[col_j]
                        dx = col_j - col_i  # Distance in grid coordinates
                        
                        # Bidirectional edges
                        mesh_src.extend([idx_i, idx_j])
                        mesh_dst.extend([idx_j, idx_i])
                        mesh_distances.extend([dx, -dx])
                    
                    if mesh_src:
                        mesh_edge_index = torch.tensor([mesh_src, mesh_dst], dtype=torch.long)
                        edge_index = torch.cat([edge_index, mesh_edge_index], dim=1)
                        
                        # Edge attributes for mesh edges with inverse distance
                        num_mesh = len(mesh_src)
                        mesh_attr_list = []
                        for i in range(num_mesh):
                            d = mesh_distances[i]
                            # Row connections have dy, dx=0; Col connections have dx, dy=0
                            # Determine which type based on whether we're in row section or col section
                            if i < 2 * (len(rows) - 1):  # Row mesh connections
                                inv_dx = 0.0
                                inv_dy = (1.0 if d > 0 else -1.0) / (abs(d) + 1e-6)
                            else:  # Col mesh connections
                                inv_dx = (1.0 if d > 0 else -1.0) / (abs(d) + 1e-6)
                                inv_dy = 0.0
                            
                            features = [inv_dx, inv_dy, 0.0]  # [inv_dx, inv_dy, is_meta=0]
                            if self.use_conflict_edges:
                                features.append(0.0)  # is_conflict=0
                            if self.use_meta_mesh:
                                features.append(1.0)  # is_meta_mesh=1 (this IS a mesh edge)
                            if self.use_meta_row_col_edges:
                                features.append(0.0)  # is_meta_row_col_cross=0
                            if self.use_edge_labels_as_features:
                                features.extend([0.0, 0.0])  # [bridge_label=0, is_labeled=0]
                            mesh_attr_list.append(features)
                        
                        mesh_edge_attr = torch.tensor(mesh_attr_list, dtype=torch.float)
                        edge_attr = torch.cat([edge_attr, mesh_edge_attr], dim=0)
                        
                        # Update mask
                        edge_mask = torch.cat([edge_mask, torch.zeros(num_mesh, dtype=torch.bool)])
                
                # 4b. Row-Col Cross Edges: Connect each row meta to each col meta
                if self.use_meta_row_col_edges:
                    cross_src = []
                    cross_dst = []
                    
                    # Create all pairs of row-col connections (bidirectional)
                    for r in rows:
                        for c in cols:
                            row_idx = row_map[r]
                            col_idx = col_map[c]
                            cross_src.extend([row_idx, col_idx])
                            cross_dst.extend([col_idx, row_idx])
                    
                    if cross_src:
                        cross_edge_index = torch.tensor([cross_src, cross_dst], dtype=torch.long)
                        edge_index = torch.cat([edge_index, cross_edge_index], dim=1)
                        
                        # Edge attributes for cross edges: dx=dy=0
                        num_cross = len(cross_src)
                        cross_attr_list = []
                        for _ in range(num_cross):
                            features = [0.0, 0.0, 0.0]  # [inv_dx=0, inv_dy=0, is_meta=0]
                            if self.use_conflict_edges:
                                features.append(0.0)  # is_conflict=0
                            if self.use_meta_mesh:
                                features.append(0.0)  # is_meta_mesh=0
                            if self.use_meta_row_col_edges:
                                features.append(1.0)  # is_meta_row_col_cross=1 (this IS a cross edge)
                            if self.use_edge_labels_as_features:
                                features.extend([0.0, 0.0])  # [bridge_label=0, is_labeled=0]
                            cross_attr_list.append(features)
                        
                        cross_edge_attr = torch.tensor(cross_attr_list, dtype=torch.float)
                        edge_attr = torch.cat([edge_attr, cross_edge_attr], dim=0)
                        
                        # Update mask
                        edge_mask = torch.cat([edge_mask, torch.zeros(num_cross, dtype=torch.bool)])
                
                # 4c. Global Meta ↔ Row/Col Meta Edges (if both global and row/col metas are enabled)
                if self.use_meta_node:
                    # Connect global meta to all row and col meta nodes
                    global_rc_src = []
                    global_rc_dst = []
                    
                    # Get all row/col meta indices
                    all_line_meta_indices = list(row_map.values()) + list(col_map.values())
                    
                    for line_meta_idx in all_line_meta_indices:
                        # Bidirectional: global ↔ line meta
                        global_rc_src.extend([meta_idx, line_meta_idx])
                        global_rc_dst.extend([line_meta_idx, meta_idx])
                    
                    if global_rc_src:
                        global_rc_edge_index = torch.tensor([global_rc_src, global_rc_dst], dtype=torch.long)
                        edge_index = torch.cat([edge_index, global_rc_edge_index], dim=1)
                        
                        # Edge attributes: is_meta=1 (meta-to-meta edge)
                        num_global_rc = len(global_rc_src)
                        global_rc_attr_list = []
                        for _ in range(num_global_rc):
                            features = [0.0, 0.0, 1.0]  # [inv_dx=0, inv_dy=0, is_meta=1]
                            if self.use_conflict_edges:
                                features.append(0.0)  # is_conflict=0
                            if self.use_meta_mesh:
                                features.append(0.0)  # is_meta_mesh=0
                            if self.use_meta_row_col_edges:
                                features.append(0.0)  # is_meta_row_col_cross=0 (this is global-to-line, not row-to-col)
                            if self.use_edge_labels_as_features:
                                features.extend([0.0, 0.0])  # [bridge_label=0, is_labeled=0]
                            global_rc_attr_list.append(features)
                        
                        global_rc_edge_attr = torch.tensor(global_rc_attr_list, dtype=torch.float)
                        edge_attr = torch.cat([edge_attr, global_rc_edge_attr], dim=0)
                        
                        # Update mask (these are not prediction targets)
                        edge_mask = torch.cat([edge_mask, torch.zeros(num_global_rc, dtype=torch.bool)])
            
            # If no edge features were requested but code expects edge_attr to exist
            if edge_attr.numel() == 0:
                edge_attr = None

            # 5. Extract edge conflict indices for crossing loss
            edge_conflict_indices = []
            if 'edge_conflicts' in graph_info:
                # Create mapping from (source, target) to edge index in tensor
                # We need to map both forward and backward edges
                edge_map = {}
                for idx, (src_id, tgt_id) in enumerate(zip(source_nodes, target_nodes)):
                    edge_map[(src_id, tgt_id)] = idx  # forward edge
                    edge_map[(tgt_id, src_id)] = idx + len(source_nodes)  # backward edge
                
                for conflict in graph_info['edge_conflicts']:
                    e1_src = conflict['edge1']['source']
                    e1_tgt = conflict['edge1']['target']
                    e2_src = conflict['edge2']['source']
                    e2_tgt = conflict['edge2']['target']
                    
                    # Find edge indices for both edges
                    e1_idx = edge_map.get((e1_src, e1_tgt)) or edge_map.get((e1_tgt, e1_src))
                    e2_idx = edge_map.get((e2_src, e2_tgt)) or edge_map.get((e2_tgt, e2_src))
                    
                    if e1_idx is not None and e2_idx is not None:
                        # Store both (e1, e2) and (e2, e1) for symmetry
                        edge_conflict_indices.append((e1_idx, e2_idx))
                        # Also add reverse direction conflicts
                        e1_idx_rev = edge_map.get((e1_tgt, e1_src), e1_idx)
                        e2_idx_rev = edge_map.get((e2_tgt, e2_src), e2_idx)
                        if e1_idx_rev != e1_idx or e2_idx_rev != e2_idx:
                            edge_conflict_indices.append((e1_idx_rev, e2_idx_rev))

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                       edge_mask=edge_mask, edge_conflicts=edge_conflict_indices)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, Path(self.processed_dir) / processed_filename)
