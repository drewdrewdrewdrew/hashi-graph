"""Custom PyTorch Geometric dataset for Hashi puzzle graphs."""

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import networkx as nx
import numpy as np
import scipy.sparse.linalg
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

# Node Type Constants
NODE_TYPE_ISLAND_START = 1
NODE_TYPE_ISLAND_END = 8
NODE_TYPE_GLOBAL_META = 9
NODE_TYPE_ROW_COL_META = 10
NODE_TYPE_COMPONENT_META = 11


class FeatureSchema:
    """
    Centralized schema for node and edge feature indices.

    This class maps feature names to their respective indices in the feature tensors,
    allowing code to avoid hardcoded indices like x[:, 0].
    """

    def __init__(self, node_map: dict[str, int], edge_map: dict[str, int]):
        self.node_map = node_map
        self.edge_map = edge_map

    def get_node_idx(self, name: str) -> int:
        """Get the index of a node feature by name."""
        if name not in self.node_map:
            available = list(self.node_map.keys())
            msg = f"Node feature '{name}' not found in schema. Available: {available}"
            raise ValueError(msg)
        return self.node_map[name]

    def get_edge_idx(self, name: str) -> int:
        """Get the index of an edge feature by name."""
        if name not in self.edge_map:
            available = list(self.edge_map.keys())
            msg = f"Edge feature '{name}' not found in schema. Available: {available}"
            raise ValueError(msg)
        return self.edge_map[name]


class HashiDatasetCache:
    """In-memory singleton cache for processed Hashi datasets."""

    _cache: ClassVar[dict[str, "HashiDataset"]] = {}

    @classmethod
    def _config_hash(cls, config: dict[str, Any], split: str) -> str:
        """Create a stable hash of the relevant data/model configuration."""
        # Extract only features that affect dataset processing
        data_keys = ["size", "difficulty", "limit"]
        model_keys = [
            "use_degree",
            "use_global_meta_node",
            "use_row_col_meta",
            "use_meta_mesh",
            "use_meta_row_col_edges",
            "use_distance",
            "use_edge_labels_as_features",
            "use_closeness_centrality",
            "use_conflict_edges",
            "use_capacity",
            "use_structural_degree",
            "use_structural_degree_nsew",
            "use_unused_capacity",
            "use_conflict_status",
            "use_articulation_points",
            "use_cut_edges",
            "use_spectral_features",
            "use_potential_crossing",
            "use_component_meta",
        ]

        relevant_config = {
            "split": split,
            "data": {k: config["data"].get(k) for k in data_keys},
            "model": {k: config["model"].get(k) for k in model_keys},
        }

        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    @classmethod
    def get_or_create(
        cls,
        config: dict[str, Any],
        split: str,
        transform: Callable[[Data], Data] | None = None,
    ) -> "HashiDataset":
        """Get a dataset from cache or create a new one."""
        key = cls._config_hash(config, split)
        if key not in cls._cache:
            data_config = config["data"]
            model_config = config["model"]
            # We import here to avoid circular dependency if HashiDataset uses the cache
            from .data import HashiDataset

            cls._cache[key] = HashiDataset(
                root=data_config["root_dir"],
                split=split,
                size=data_config.get("size"),
                difficulty=data_config.get("difficulty"),
                limit=data_config.get("limit"),
                use_degree=model_config.get("use_degree", False),
                use_meta_node=model_config.get("use_global_meta_node", True),
                use_row_col_meta=model_config.get("use_row_col_meta", False),
                use_meta_mesh=model_config.get("use_meta_mesh", False),
                use_meta_row_col_edges=model_config.get(
                    "use_meta_row_col_edges", False,
                ),
                use_distance=model_config.get("use_distance", False),
                use_edge_labels_as_features=model_config.get(
                    "use_edge_labels_as_features", False,
                ),
                use_closeness_centrality=model_config.get(
                    "use_closeness_centrality", False,
                ),
                use_conflict_edges=model_config.get("use_conflict_edges", False),
                use_capacity=model_config.get("use_capacity", True),
                use_structural_degree=model_config.get("use_structural_degree", True),
                use_structural_degree_nsew=model_config.get(
                    "use_structural_degree_nsew", False,
                ),
                use_unused_capacity=model_config.get("use_unused_capacity", True),
                use_conflict_status=model_config.get("use_conflict_status", True),
                use_articulation_points=model_config.get(
                    "use_articulation_points", False,
                ),
                use_cut_edges=model_config.get("use_cut_edges", False),
                use_spectral_features=model_config.get("use_spectral_features", False),
                use_potential_crossing=model_config.get("use_potential_crossing", False),
                use_component_meta=model_config.get("use_component_meta", False),
                transform=transform,
            )
        return cls._cache[key]


class MakeBidirectional:
    """
    Reconstruct reverse edges from a graph that only stores one direction.

    Also reconstructs symmetric edge attributes (negating dx/dy for
    reverse edges).
    """

    def __call__(self, data: Data) -> Data:
        """Transform the data by making edges bidirectional."""
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        row, col = data.edge_index

        # 1. Create reverse edges
        rev_edge_index = torch.stack([col, row], dim=0)
        data.edge_index = torch.cat([data.edge_index, rev_edge_index], dim=1)

        # 2. Duplicate labels (y)
        if hasattr(data, "y") and data.y is not None:
            data.y = torch.cat([data.y, data.y], dim=0)

        # 3. Duplicate edge_mask
        if hasattr(data, "edge_mask") and data.edge_mask is not None:
            data.edge_mask = torch.cat([data.edge_mask, data.edge_mask], dim=0)

        # 4. Handle edge_attr (negate only inv_dx and inv_dy)
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            fwd_attr = data.edge_attr
            rev_attr = fwd_attr.clone()

            # Negate inv_dx (idx 0) and inv_dy (idx 1)
            # Only if they are non-zero (to be safe)
            # Actually simple negation is fine for 0 too
            rev_attr[:, 0] *= -1
            rev_attr[:, 1] *= -1

            data.edge_attr = torch.cat([fwd_attr, rev_attr], dim=0)

        return data


class GridStretch:
    """
    Randomly inserts an empty gap row or column into the puzzle to vary distances.

    Updates pos and recalculates inv_dx/inv_dy features.
    """

    def __init__(self, prob: float = 0.5, max_gap: int = 3) -> None:
        self.prob = prob
        self.max_gap = max_gap

    def __call__(self, data: Data) -> Data:
        """Randomly stretch the grid by inserting gaps."""
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
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            row, col = data.edge_index
            diffs = data.pos[col] - data.pos[row]

            # Recalculate inv_dx, inv_dy
            # sign(d) / (|d| + eps)
            new_inv_dx = torch.sign(diffs[:, 0]) / (torch.abs(diffs[:, 0]) + 1e-6)
            new_inv_dy = torch.sign(diffs[:, 1]) / (torch.abs(diffs[:, 1]) + 1e-6)

            mask_dx = torch.abs(data.edge_attr[:, 0]) > 1e-6
            mask_dy = torch.abs(data.edge_attr[:, 1]) > 1e-6

            data.edge_attr[mask_dx, 0] = new_inv_dx[mask_dx]
            data.edge_attr[mask_dy, 1] = new_inv_dy[mask_dy]

        return data


class RandomHashiAugment:
    """Composes geometric augmentations: Rotate, Flip, Stretch."""

    def __init__(self, stretch_prob: float = 0.5, max_stretch: int = 3) -> None:
        self.stretch = GridStretch(prob=stretch_prob, max_gap=max_stretch)

    def __call__(self, data: Data) -> Data:
        """Apply random geometric augmentations to the data."""
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
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
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

    def __init__(
        self,
        root: str,
        split: str = "train",
        size: list[int] | None = None,
        difficulty: list[int] | None = None,
        limit: int | None = None,
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
        use_articulation_points: bool = False,
        use_cut_edges: bool = False,
        use_spectral_features: bool = False,
        use_potential_crossing: bool = False,
        use_component_meta: bool = False,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
    ) -> None:
        """
        Initialize the HashiDataset.

        Args:
            root (str): Root directory where the dataset should be stored.
            split (str): The dataset split, one of 'train', 'val', or 'test'.
            size (Optional[List[int]]): List of puzzle sizes to include.
            difficulty (Optional[List[int]]): List of difficulties to include.
            limit (Optional[int]): Limit the dataset to the first `limit` files.
            use_degree (bool): Whether to include node degree as a feature.
            Default: False.
            use_meta_node (bool): Whether to add a meta node connected to all other
            nodes.
            Default: False.
            use_row_col_meta (bool): Whether to add row/column meta nodes.
            Default: False.
            use_meta_mesh (bool): Whether to connect row metas to each other and col
            metas
            to each other.
            Default: False.
            use_meta_row_col_edges (bool): Whether to connect each row meta to each col
            meta.
            Default: False.
            use_distance (bool): Whether to include inverse signed distance as an edge
            feature.
            Default: False.
            use_edge_labels_as_features (bool): Whether to include edge labels as input
            features for masking.
            Default: False.
            use_closeness_centrality (bool): Whether to include closeness centrality as
            a node
            feature.
            Default: False.
            use_conflict_edges (bool): Whether to add conflict edges for crossing
            constraints.
            Default: False.
            use_capacity (bool): Whether to include logical capacity as a node feature.
            Default: True.
            use_structural_degree (bool): Whether to include structural degree count
            (1-4)
            as a node feature.
            Default: True.
            use_structural_degree_nsew (bool): Whether to include structural degree as
            NSEW
            bitmask (0-15) as a node feature.
            Default: False.
            use_unused_capacity (bool): Whether to include unused capacity as a node
            feature.
            Default: True.
            use_conflict_status (bool): Whether to include conflict status as a node
            feature.
            Default: True.
            use_articulation_points (bool): Whether to include articulation points as a
            node feature. Default: False.
            use_cut_edges (bool): Whether to include cut edges (bridges) as an edge
            feature. Default: False.
            use_spectral_features (bool): Whether to include spectral fingerprinting
            (3 eigenvectors) as a node feature. Default: False.
            use_component_meta (bool): Whether to include component meta nodes.
            Default: False.
            transform (callable, optional): A function/transform for the data object.
            pre_transform (callable, optional): A function/transform for the data object
            before saving.
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
        self.use_articulation_points = use_articulation_points
        self.use_cut_edges = use_cut_edges
        self.use_spectral_features = use_spectral_features
        self.use_potential_crossing = use_potential_crossing
        self.use_component_meta = use_component_meta

        # We must determine the raw file names before calling super().__init__()
        # so the parent class can correctly check if processing is needed.
        self._raw_filenames = self._get_filtered_filenames(root)

        # Instantiate bidirectional transform for use in get()
        self.make_bidirectional = MakeBidirectional()

        super().__init__(root, transform, pre_transform)

    @property
    def processed_dir(self) -> str:
        """Override to make processed directory config-dependent."""
        # Create a hash of config parameters that affect dataset processing
        config_params = {
            "use_degree": self.use_degree,
            "use_meta_node": self.use_meta_node,
            "use_row_col_meta": self.use_row_col_meta,
            "use_meta_mesh": self.use_meta_mesh,
            "use_meta_row_col_edges": self.use_meta_row_col_edges,
            "use_distance": self.use_distance,
            "use_edge_labels_as_features": self.use_edge_labels_as_features,
            "use_closeness_centrality": self.use_closeness_centrality,
            "use_conflict_edges": self.use_conflict_edges,
            "use_capacity": self.use_capacity,
            "use_structural_degree": self.use_structural_degree,
            "use_structural_degree_nsew": self.use_structural_degree_nsew,
            "use_unused_capacity": self.use_unused_capacity,
            "use_conflict_status": self.use_conflict_status,
            "use_articulation_points": self.use_articulation_points,
            "use_cut_edges": self.use_cut_edges,
            "use_spectral_features": self.use_spectral_features,
            "use_potential_crossing": self.use_potential_crossing,
            "use_component_meta": self.use_component_meta,
        }
        config_str = json.dumps(config_params, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        return str(Path(self.root) / f"processed_{config_hash}")

    def _get_filtered_filenames(self, root: str) -> list[str]:
        """Scan and filter raw files based on instance attributes."""
        raw_dir = Path(root) / "raw"
        filenames = []
        if not raw_dir.is_dir():
            msg = f"Raw data directory not found at {raw_dir}"
            raise RuntimeError(msg)

        for path in raw_dir.glob("puzzle_*.json"):
            with Path(path).open() as f:
                data = json.load(f)

            if data.get("split") != self.split:
                continue
            if (
                self.size_filter
                and data["generation_params"].get("size") not in self.size_filter
            ):
                continue
            if (
                self.difficulty_filter
                and data["generation_params"].get("difficulty")
                not in self.difficulty_filter
            ):
                continue

            filenames.append(path.name)

        if not filenames:
            msg = f"No files found for split '{self.split}' with the given filters."
            raise RuntimeError(msg)

        # Sort for reproducibility and apply limit
        filenames = sorted(filenames)
        if self.limit is not None:
            filenames = filenames[: int(self.limit)]

        return filenames

    @property
    def raw_file_names(self) -> list[str]:
        """Return the list of raw file names."""
        return self._raw_filenames

    @property
    def processed_file_names(self) -> list[str]:
        """Return the list of processed file names."""
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
        if self.use_articulation_points:
            suffix += "_ap"
        if self.use_cut_edges:
            suffix += "_cut"
        if self.use_spectral_features:
            suffix += "_spec"
        if self.use_potential_crossing:
            suffix += "_cross"
        if self.use_component_meta:
            suffix += "_comp"

        # Add _oneway suffix to distinguish from old bidirectional files
        suffix += "_oneway"

        return [f"{Path(fn).stem}{suffix}.pt" for fn in self._raw_filenames]

    def _get_feature_schema(self) -> FeatureSchema:
        """Compute the feature schema based on the current dataset configuration."""
        node_map = {}
        current_idx = 0

        if self.use_capacity:
            node_map["capacity"] = current_idx
            current_idx += 1
        if self.use_structural_degree or self.use_structural_degree_nsew:
            node_map["structural_degree"] = current_idx
            current_idx += 1
        if self.use_unused_capacity:
            node_map["unused_capacity"] = current_idx
            current_idx += 1
        if self.use_conflict_status:
            node_map["conflict_status"] = current_idx
            current_idx += 1
        if self.use_closeness_centrality:
            node_map["closeness_centrality"] = current_idx
            current_idx += 1
        if self.use_articulation_points:
            node_map["articulation_point"] = current_idx
            current_idx += 1
        if self.use_spectral_features:
            node_map["spectral_1"] = current_idx
            node_map["spectral_2"] = current_idx + 1
            node_map["spectral_3"] = current_idx + 2
            current_idx += 3

        edge_map = {}
        current_idx = 0
        # Base: inv_dx, inv_dy, is_meta
        edge_map["inv_dx"] = current_idx
        edge_map["inv_dy"] = current_idx + 1
        edge_map["is_meta"] = current_idx + 2
        current_idx += 3

        if self.use_conflict_edges:
            edge_map["is_conflict"] = current_idx
            current_idx += 1
        if self.use_meta_mesh:
            edge_map["is_meta_mesh"] = current_idx
            current_idx += 1
        if self.use_meta_row_col_edges:
            edge_map["is_meta_row_col_cross"] = current_idx
            current_idx += 1
        if self.use_edge_labels_as_features:
            edge_map["bridge_label"] = current_idx
            edge_map["is_labeled"] = current_idx + 1
            current_idx += 2
        if self.use_cut_edges:
            edge_map["is_cut_edge"] = current_idx
            current_idx += 1
        if self.use_potential_crossing:
            edge_map["is_potential_crossing"] = current_idx
            current_idx += 1

        return FeatureSchema(node_map, edge_map)

    def len(self) -> int:
        """Return the number of data points."""
        return len(self._raw_filenames)

    def get(self, idx: int) -> Data:
        """Get the data object at index `idx`."""
        processed_filename = self.processed_file_names[idx]
        # Set weights_only=False to allow loading Data objects.
        data = torch.load(
            Path(self.processed_dir) / processed_filename, weights_only=False,
        )

        # Apply MakeBidirectional to reconstruct the full graph on the fly
        data = self.make_bidirectional(data)

        # Ensure edge_mask exists (for backward compatibility)
        if not hasattr(data, "edge_mask") or data.edge_mask is None:
            # All edges are original if no mask exists
            data.edge_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)

        # Ensure edge_conflicts exists (for backward compatibility)
        if not hasattr(data, "edge_conflicts") or data.edge_conflicts is None:
            data.edge_conflicts = []

        return data

    def _build_island_nodes(
        self,
        graph_info: dict[str, Any],
        schema: FeatureSchema,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[list[float]],
        dict[int, int],
        set[tuple[int, int]],
    ]:
        """Process core puzzle nodes (islands)."""
        node_features = []
        node_type_list = []
        node_pos_list = [node["pos"] for node in graph_info["nodes"]]
        num_nodes = len(graph_info["nodes"])
        node_id_to_idx = {node["id"]: i for i, node in enumerate(graph_info["nodes"])}

        # Pre-compute structural degrees (O(N) with spatial hash)
        structural_degrees = {}
        pos_to_id = {(n["pos"][0], n["pos"][1]): n["id"] for n in graph_info["nodes"]}
        for node in graph_info["nodes"]:
            x_pos, y_pos = node["pos"]
            north = (x_pos, y_pos - 1) in pos_to_id
            south = (x_pos, y_pos + 1) in pos_to_id
            west = (x_pos - 1, y_pos) in pos_to_id
            east = (x_pos + 1, y_pos) in pos_to_id

            if self.use_structural_degree_nsew:
                bitmask = (north * 1) | (south * 2) | (west * 4) | (east * 8)
                structural_degrees[node["id"]] = max(1, bitmask)
            else:
                degree = north + south + west + east
                structural_degrees[node["id"]] = max(1, degree)

        # Pre-compute conflict status
        conflict_nodes = set()
        if "edge_conflicts" in graph_info:
            for conflict in graph_info["edge_conflicts"]:
                conflict_nodes.add(conflict["edge1"]["source"])
                conflict_nodes.add(conflict["edge1"]["target"])
                conflict_nodes.add(conflict["edge2"]["source"])
                conflict_nodes.add(conflict["edge2"]["target"])

        # Articulation points and bridges
        articulation_points = set()
        bridges = set()
        spectral_features = {}
        if (
            self.use_articulation_points
            or self.use_cut_edges
            or self.use_spectral_features
        ):
            g_potential = nx.Graph()
            g_potential.add_nodes_from(range(num_nodes))
            for edge in graph_info["edges"]:
                u = node_id_to_idx[edge["source"]]
                v = node_id_to_idx[edge["target"]]
                g_potential.add_edge(u, v)

            if self.use_articulation_points:
                articulation_points = set(nx.articulation_points(g_potential))

            if self.use_cut_edges:
                bridges = {
                    tuple(sorted((u, v))) for u, v in nx.bridges(g_potential)
                }

            if self.use_spectral_features:
                try:
                    k = 3
                    if g_potential.number_of_nodes() > k + 1:
                        l_matrix = nx.normalized_laplacian_matrix(g_potential)
                        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                            l_matrix, k=k + 1, which="SM", maxiter=1000,
                        )
                        idx = eigenvalues.argsort()
                        vectors = eigenvectors[:, idx][:, 1 : k + 1]
                        for i in range(vectors.shape[1]):
                            col = vectors[:, i]
                            max_val = np.abs(col).max()
                            if max_val > 1e-9:
                                col = col / max_val
                            vectors[:, i] = col
                        for i in range(num_nodes):
                            spectral_features[i] = vectors[i].tolist()
                    else:
                        for i in range(num_nodes):
                            spectral_features[i] = [0.0] * k
                except Exception as e:
                    print(f"Warning: Spectral feature computation failed: {e}")
                    for i in range(num_nodes):
                        spectral_features[i] = [0.0] * 3

        # Build feature vectors
        num_feats = len(schema.node_map)
        for i, node in enumerate(graph_info["nodes"]):
            features = [0.0] * num_feats
            node_type_list.append(node["n"])

            if self.use_capacity:
                features[schema.get_node_idx("capacity")] = float(node["n"])
            if self.use_structural_degree or self.use_structural_degree_nsew:
                features[schema.get_node_idx("structural_degree")] = float(
                    structural_degrees[node["id"]],
                )
            if self.use_unused_capacity:
                features[schema.get_node_idx("unused_capacity")] = float(node["n"])
            if self.use_conflict_status:
                features[schema.get_node_idx("conflict_status")] = (
                    1.0 if node["id"] in conflict_nodes else 0.0
                )
            if self.use_closeness_centrality:
                features[schema.get_node_idx("closeness_centrality")] = float(
                    node.get("closeness_centrality", 0.0),
                )
            if self.use_articulation_points:
                features[schema.get_node_idx("articulation_point")] = (
                    1.0 if i in articulation_points else 0.0
                )
            if self.use_spectral_features:
                spec = spectral_features.get(i, [0.0, 0.0, 0.0])
                features[schema.get_node_idx("spectral_1")] = spec[0]
                features[schema.get_node_idx("spectral_2")] = spec[1]
                features[schema.get_node_idx("spectral_3")] = spec[2]

            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)
        node_type = torch.tensor(node_type_list, dtype=torch.long)
        return x, node_type, node_pos_list, node_id_to_idx, bridges

    def _build_meta_nodes(
        self,
        graph_info: dict[str, Any],
        schema: FeatureSchema,
        x: torch.Tensor,
        node_type: torch.Tensor,
        node_pos_list: list[list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[float]], dict[str, Any]]:
        """Handle Global and Row/Col metas."""
        num_feats = len(schema.node_map)
        meta_info = {}

        # 1. Global Meta Node
        if self.use_meta_node:
            meta_feat = [0.0] * num_feats
            if self.use_capacity:
                meta_feat[schema.get_node_idx("capacity")] = float(
                    NODE_TYPE_GLOBAL_META,
                )

            global_idx = x.size(0)
            x = torch.cat([x, torch.tensor([meta_feat], dtype=torch.float)], dim=0)
            node_type = torch.cat(
                [node_type, torch.tensor([NODE_TYPE_GLOBAL_META], dtype=torch.long)],
                dim=0,
            )
            node_pos_list.append([-1000.0, -1000.0])
            meta_info["global_idx"] = global_idx

        # 2. Row/Col Meta Nodes
        if self.use_row_col_meta:
            rows = sorted({n["pos"][1] for n in graph_info["nodes"]})
            cols = sorted({n["pos"][0] for n in graph_info["nodes"]})

            row_map = {}
            row_feats = []
            for r in rows:
                row_map[r] = x.size(0) + len(row_feats)
                feat = [0.0] * num_feats
                if self.use_capacity:
                    feat[schema.get_node_idx("capacity")] = float(
                        NODE_TYPE_ROW_COL_META,
                    )
                row_feats.append(feat)
                node_pos_list.append([-1000.0, float(r)])

            x = torch.cat([x, torch.tensor(row_feats, dtype=torch.float)], dim=0)
            node_type = torch.cat(
                [
                    node_type,
                    torch.full(
                        (len(rows),), NODE_TYPE_ROW_COL_META, dtype=torch.long,
                    ),
                ],
                dim=0,
            )

            col_map = {}
            col_feats = []
            for c in cols:
                col_map[c] = x.size(0) + len(col_feats)
                feat = [0.0] * num_feats
                if self.use_capacity:
                    feat[schema.get_node_idx("capacity")] = float(
                        NODE_TYPE_ROW_COL_META,
                    )
                col_feats.append(feat)
                node_pos_list.append([float(c), -1000.0])

            x = torch.cat([x, torch.tensor(col_feats, dtype=torch.float)], dim=0)
            node_type = torch.cat(
                [
                    node_type,
                    torch.full(
                        (len(cols),), NODE_TYPE_ROW_COL_META, dtype=torch.long,
                    ),
                ],
                dim=0,
            )

            meta_info["rows"] = rows
            meta_info["cols"] = cols
            meta_info["row_map"] = row_map
            meta_info["col_map"] = col_map

        return x, node_type, node_pos_list, meta_info

    def _build_component_meta_nodes(
        self,
        graph_info: dict[str, Any],
        schema: FeatureSchema,
        x: torch.Tensor,
        node_type: torch.Tensor,
        node_pos_list: list[list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[float]], list[int]]:
        """Pre-allocate static N component meta nodes (initially disconnected)."""
        num_islands = len(graph_info["nodes"])
        num_feats = len(schema.node_map)

        comp_feats = []
        for _ in range(num_islands):
            feat = [0.0] * num_feats
            if self.use_capacity:
                feat[schema.get_node_idx("capacity")] = float(
                    NODE_TYPE_COMPONENT_META,
                )
            comp_feats.append(feat)
            node_pos_list.append([-2000.0, -2000.0])  # Sentinel for component meta

        start_idx = x.size(0)
        x = torch.cat([x, torch.tensor(comp_feats, dtype=torch.float)], dim=0)
        node_type = torch.cat(
            [
                node_type,
                torch.full(
                    (num_islands,), NODE_TYPE_COMPONENT_META, dtype=torch.long,
                ),
            ],
            dim=0,
        )

        comp_indices = list(range(start_idx, start_idx + num_islands))
        return x, node_type, node_pos_list, comp_indices

    def _build_edges(
        self,
        graph_info: dict[str, Any],
        schema: FeatureSchema,
        node_id_to_idx: dict[int, int],
        node_pos_list: list[list[float]],
        meta_info: dict[str, Any],
        bridges: set[tuple[int, int]] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[tuple[int, int]],
    ]:
        """Construct edge indices, attributes, labels, and mask."""
        if bridges is None:
            bridges = set()
        edge_indices = []
        edge_attrs = []
        edge_labels = []
        edge_mask_list = []
        num_edge_feats = len(schema.edge_map)

        # Pre-identify edges involved in potential crossings
        crossing_edges = set()
        if self.use_potential_crossing and "edge_conflicts" in graph_info:
            for conflict in graph_info["edge_conflicts"]:
                # Edge 1
                e1_s = node_id_to_idx[conflict["edge1"]["source"]]
                e1_t = node_id_to_idx[conflict["edge1"]["target"]]
                crossing_edges.add(tuple(sorted((e1_s, e1_t))))
                # Edge 2
                e2_s = node_id_to_idx[conflict["edge2"]["source"]]
                e2_t = node_id_to_idx[conflict["edge2"]["target"]]
                crossing_edges.add(tuple(sorted((e2_s, e2_t))))

        # 1. Original Puzzle Edges
        puzzle_edge_ids = []
        for edge in graph_info["edges"]:
            s = node_id_to_idx[edge["source"]]
            t = node_id_to_idx[edge["target"]]
            label = edge["label"]

            # Store canonical source < target
            if s > t:
                s, t = t, s

            puzzle_edge_ids.append((s, t))
            edge_indices.append([s, t])
            edge_labels.append(label)
            edge_mask_list.append(True)

            # Features
            p1 = node_pos_list[s]
            p2 = node_pos_list[t]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]

            feat = [0.0] * num_edge_feats
            feat[schema.get_edge_idx("inv_dx")] = (
                (1.0 if dx > 0 else -1.0) / (abs(dx) + 1e-6) if abs(dx) > 1e-6 else 0.0
            )
            feat[schema.get_edge_idx("inv_dy")] = (
                (1.0 if dy > 0 else -1.0) / (abs(dy) + 1e-6) if abs(dy) > 1e-6 else 0.0
            )
            feat[schema.get_edge_idx("is_meta")] = 0.0

            if self.use_edge_labels_as_features:
                feat[schema.get_edge_idx("bridge_label")] = float(label)
                feat[schema.get_edge_idx("is_labeled")] = 1.0

            if self.use_cut_edges:
                is_bridge = (
                    1.0 if tuple(sorted((s, t))) in bridges else 0.0
                )
                feat[schema.get_edge_idx("is_cut_edge")] = is_bridge

            if self.use_potential_crossing:
                is_potential = (
                    1.0 if tuple(sorted((s, t))) in crossing_edges else 0.0
                )
                feat[schema.get_edge_idx("is_potential_crossing")] = is_potential

            edge_attrs.append(feat)

        # 2. Conflict Edges
        if self.use_conflict_edges and "edge_conflicts" in graph_info:
            for conflict in graph_info["edge_conflicts"]:
                e1 = conflict["edge1"]
                e2 = conflict["edge2"]
                for n1_id in [e1["source"], e1["target"]]:
                    for n2_id in [e2["source"], e2["target"]]:
                        s = node_id_to_idx[n1_id]
                        t = node_id_to_idx[n2_id]
                        if s > t:
                            s, t = t, s

                        edge_indices.append([s, t])
                        edge_labels.append(0)
                        edge_mask_list.append(False)

                        feat = [0.0] * num_edge_feats
                        feat[schema.get_edge_idx("is_meta")] = 0.0
                        feat[schema.get_edge_idx("is_conflict")] = 1.0
                        edge_attrs.append(feat)

        # 3. Global Meta Edges
        if self.use_meta_node and "global_idx" in meta_info:
            g_idx = meta_info["global_idx"]
            num_islands = len(graph_info["nodes"])
            for i in range(num_islands):
                edge_indices.append([i, g_idx])
                edge_labels.append(0)
                edge_mask_list.append(False)

                feat = [0.0] * num_edge_feats
                feat[schema.get_edge_idx("is_meta")] = 1.0
                edge_attrs.append(feat)

        # 4. Row/Col Meta Edges
        if self.use_row_col_meta:
            row_map = meta_info["row_map"]
            col_map = meta_info["col_map"]
            for i, node in enumerate(graph_info["nodes"]):
                # Node -> Row Meta
                r_idx = row_map[node["pos"][1]]
                edge_indices.append([i, r_idx])
                edge_labels.append(0)
                edge_mask_list.append(False)

                feat = [0.0] * num_edge_feats
                feat[schema.get_edge_idx("is_meta")] = 1.0
                edge_attrs.append(feat)

                # Node -> Col Meta
                c_idx = col_map[node["pos"][0]]
                edge_indices.append([i, c_idx])
                edge_labels.append(0)
                edge_mask_list.append(False)

                feat = [0.0] * num_edge_feats
                feat[schema.get_edge_idx("is_meta")] = 1.0
                edge_attrs.append(feat)

            # Meta Mesh
            if self.use_meta_mesh:
                rows = meta_info["rows"]
                cols = meta_info["cols"]
                for i in range(len(rows) - 1):
                    s, t = row_map[rows[i]], row_map[rows[i + 1]]
                    dy = rows[i + 1] - rows[i]
                    edge_indices.append([s, t])
                    edge_labels.append(0)
                    edge_mask_list.append(False)
                    feat = [0.0] * num_edge_feats
                    feat[schema.get_edge_idx("is_meta_mesh")] = 1.0
                    feat[schema.get_edge_idx("inv_dy")] = 1.0 / (dy + 1e-6)
                    edge_attrs.append(feat)

                for i in range(len(cols) - 1):
                    s, t = col_map[cols[i]], col_map[cols[i + 1]]
                    dx = cols[i + 1] - cols[i]
                    edge_indices.append([s, t])
                    edge_labels.append(0)
                    edge_mask_list.append(False)
                    feat = [0.0] * num_edge_feats
                    feat[schema.get_edge_idx("is_meta_mesh")] = 1.0
                    feat[schema.get_edge_idx("inv_dx")] = 1.0 / (dx + 1e-6)
                    edge_attrs.append(feat)

            # Row-Col Cross
            if self.use_meta_row_col_edges:
                for r_idx in row_map.values():
                    for c_idx in col_map.values():
                        edge_indices.append([r_idx, c_idx])
                        edge_labels.append(0)
                        edge_mask_list.append(False)
                        feat = [0.0] * num_edge_feats
                        feat[schema.get_edge_idx("is_meta_row_col_cross")] = 1.0
                        edge_attrs.append(feat)

            # Global Meta <-> Row/Col Meta
            if self.use_meta_node:
                g_idx = meta_info["global_idx"]
                for line_idx in list(row_map.values()) + list(col_map.values()):
                    edge_indices.append([g_idx, line_idx])
                    edge_labels.append(0)
                    edge_mask_list.append(False)
                    feat = [0.0] * num_edge_feats
                    feat[schema.get_edge_idx("is_meta")] = 1.0
                    edge_attrs.append(feat)

        # 5. Component Meta Edges (Initially island i connects to meta N+i)
        if self.use_component_meta:
            num_islands = len(graph_info["nodes"])
            for i in range(num_islands):
                # Meta node index is N+i because we added them right after islands
                m_idx = num_islands + i
                edge_indices.append([i, m_idx])
                edge_labels.append(0)
                edge_mask_list.append(False)

                feat = [0.0] * num_edge_feats
                feat[schema.get_edge_idx("is_meta")] = 1.0
                edge_attrs.append(feat)

        # 6. Conflict Indices Mapping
        edge_conflict_indices = []
        if "edge_conflicts" in graph_info:
            puzzle_edge_map = {pair: i for i, pair in enumerate(puzzle_edge_ids)}
            for conflict in graph_info["edge_conflicts"]:
                e1_ids = sorted(
                    [node_id_to_idx[conflict["edge1"]["source"]],
                     node_id_to_idx[conflict["edge1"]["target"]]],
                )
                e2_ids = sorted(
                    [node_id_to_idx[conflict["edge2"]["source"]],
                     node_id_to_idx[conflict["edge2"]["target"]]],
                )
                idx1 = puzzle_edge_map.get(tuple(e1_ids))
                idx2 = puzzle_edge_map.get(tuple(e2_ids))
                if idx1 is not None and idx2 is not None:
                    edge_conflict_indices.append((idx1, idx2))
                    edge_conflict_indices.append((idx2, idx1))

        edge_index = (
            torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            if edge_indices
            else torch.empty((2, 0), dtype=torch.long)
        )
        edge_attr = (
            torch.tensor(edge_attrs, dtype=torch.float)
            if edge_attrs
            else torch.empty((0, num_edge_feats), dtype=torch.float)
        )
        y = torch.tensor(edge_labels, dtype=torch.long)
        edge_mask = torch.tensor(edge_mask_list, dtype=torch.bool)

        return edge_index, edge_attr, y, edge_mask, edge_conflict_indices

    def process(self) -> None:
        """Process raw data using modular builders and a feature schema."""
        schema = self._get_feature_schema()

        for raw_filename, processed_filename in zip(
            tqdm(self.raw_file_names, desc=f"Processing {self.split} data"),
            self.processed_file_names,
            strict=False,
        ):
            raw_path = Path(self.raw_dir) / raw_filename
            with Path(raw_path).open() as f:
                puzzle_data = json.load(f)
                graph_info = puzzle_data["graph"]

            # 1. Build Nodes (Modular)
            x, node_type, pos_list, id_to_idx, bridges = self._build_island_nodes(
                graph_info, schema,
            )
            if self.use_component_meta:
                x, node_type, pos_list, _comp_indices = (
                    self._build_component_meta_nodes(
                        graph_info, schema, x, node_type, pos_list,
                    )
                )
            x, node_type, pos_list, meta_info = self._build_meta_nodes(
                graph_info, schema, x, node_type, pos_list,
            )

            # 2. Build Edges (Modular)
            edge_index, edge_attr, y, edge_mask, edge_conflicts = self._build_edges(
                graph_info, schema, id_to_idx, pos_list, meta_info, bridges,
            )

            # 3. Create Data Object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                edge_mask=edge_mask,
                edge_conflicts=edge_conflicts,
                pos=torch.tensor(pos_list, dtype=torch.float),
                node_type=node_type,
            )

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, Path(self.processed_dir) / processed_filename)
