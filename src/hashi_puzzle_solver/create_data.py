"""
Script to create a dataset of Hashi puzzle graphs for machine learning.

- One JSON file per puzzle in dataset/raw/

Each JSON file contains:
- graph: Node-link representation with nodes and edges
- generation_params: Parameters used to generate the puzzle
- split: Train/val/test split assignment
"""
import json
import argparse
import re
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union
import random
import networkx as nx

from .graph_utils import generate_puzzle_graph


def get_existing_puzzles(raw_dir: Path) -> Tuple[int, Dict[str, int]]:
    """
    Scan existing puzzle files and return the next puzzle ID and split counts.
    
    Args:
        raw_dir: Directory containing puzzle JSON files
    
    Returns:
        Tuple of (next_puzzle_id, split_counts_dict)
    """
    if not raw_dir.exists():
        return 0, {'train': 0, 'val': 0, 'test': 0}
    
    max_id = -1
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
# Pattern for puzzle files: puzzle_00000.json
    pattern = re.compile(r'puzzle_(\d+)\.json')
    
    for file_path in raw_dir.glob('puzzle_*.json'):
        filename = file_path.name
        match = pattern.match(filename)
        if match:
            puzzle_id = int(match.group(1))
            max_id = max(max_id, puzzle_id)
            
            # Try to read the split from the JSON file
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Unable to parse puzzle file {file_path}") from exc
            split = data.get('split', 'unknown')
            if split in split_counts:
                split_counts[split] += 1
    
    next_id = max_id + 1
    return next_id, split_counts


EdgeEndpoint = Union[Tuple[int, int], Hashable]
EdgePair = Tuple[EdgeEndpoint, EdgeEndpoint]

def compute_closeness_centrality(G: nx.Graph) -> Dict[Hashable, float]:
    """
    Compute closeness centrality for all nodes on RAW graph topology.

    Important: Must be called on the RAW puzzle graph BEFORE any modifications
    like adding meta nodes, conflict edges, etc.

    Uses ALL potential edges (including label=0) to avoid data leakage.
    
    Returns: dict {node_id: closeness_value} in range [0.0, 1.0]
    """
    centrality: Dict[Hashable, float] = {}
    for node in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, node)
        if len(lengths) > 1:
            avg_distance = sum(lengths.values()) / (len(lengths) - 1)
            centrality[node] = 1.0 / avg_distance if avg_distance > 0 else 0.0
        else:
            centrality[node] = 0.0
    return centrality


def edges_cross(edge1: EdgePair, edge2: EdgePair, G: nx.Graph) -> bool:
    """
    Check if two edges would cross geometrically.
    
    Args:
        edge1, edge2: Tuples of node pairs
        G: Graph with node positions
    
    Returns:
        bool: True if edges cross
    """
    (a, b) = edge1
    (c, d) = edge2
    
    # Get positions (assuming nodes are (x,y) tuples or have x,y attributes)
    if isinstance(a, tuple):
        ax, ay = a
        bx, by = b
        cx, cy = c
        dx, dy = d
    else:
        ax, ay = G.nodes[a]['x'], G.nodes[a]['y']
        bx, by = G.nodes[b]['x'], G.nodes[b]['y']
        cx, cy = G.nodes[c]['x'], G.nodes[c]['y']
        dx, dy = G.nodes[d]['x'], G.nodes[d]['y']
    
    # Check if one is horizontal and one is vertical
    # Horizontal edge
    is_ab_horizontal = (ay == by and ax != bx)
    is_cd_horizontal = (cy == dy and cx != dx)
    
    # Vertical edge
    is_ab_vertical = (ax == bx and ay != by)
    is_cd_vertical = (cx == dx and cy != dy)
    
    # Only cross if one horizontal and one vertical
    if not ((is_ab_horizontal and is_cd_vertical) or (is_ab_vertical and is_cd_horizontal)):
        return False
    
    # Check if they actually intersect
    if is_ab_horizontal and is_cd_vertical:
        # AB is horizontal, CD is vertical
        h_y = ay  # horizontal line's y
        v_x = cx  # vertical line's x
        h_x_range = (min(ax, bx), max(ax, bx))
        v_y_range = (min(cy, dy), max(cy, dy))
        
        # Check if vertical line passes through horizontal line's y
        # and horizontal line passes through vertical line's x
        return (v_y_range[0] < h_y < v_y_range[1] and 
                h_x_range[0] < v_x < h_x_range[1])
    
    elif is_ab_vertical and is_cd_horizontal:
        # AB is vertical, CD is horizontal
        v_x = ax
        h_y = cy
        v_y_range = (min(ay, by), max(ay, by))
        h_x_range = (min(cx, dx), max(cx, dx))
        
        return (h_x_range[0] < v_x < h_x_range[1] and 
                v_y_range[0] < h_y < v_y_range[1])
    
    return False


def find_crossing_edges(G: nx.Graph) -> List[EdgePair]:
    """
    Find all pairs of edges that would geometrically cross.
    
    Args:
        G: NetworkX graph with nodes as (x,y) tuples
    
    Returns:
        List of tuples: [(edge1, edge2), ...] where edges would cross
    """
    crossings: List[EdgePair] = []
    edges = list(G.edges())
    
    for i, edge1 in enumerate(edges):
        for edge2 in edges[i+1:]:
            if edges_cross(edge1, edge2, G):
                crossings.append((edge1, edge2))
    
    return crossings


def graph_to_dict(
    G: nx.Graph,
    generation_params: Dict[str, Any],
    split: str
) -> Dict[str, Any]:
    """
    Convert a NetworkX graph to a dictionary format suitable for JSON storage.
    
    Args:
        G: NetworkX graph (RAW graph before any modifications)
        generation_params: Dictionary with generation parameters
        split: Train/val/test split assignment
    
    Returns:
        Dictionary with 'graph', 'generation_params', and 'split' keys
    """
    # Compute closeness centrality on raw graph before any modifications
    closeness_centrality = compute_closeness_centrality(G)
    
    # Create node list with sequential IDs
    nodes = []
    node_to_id = {}
    
    for idx, node in enumerate(G.nodes()):
        node_to_id[node] = idx
        if isinstance(node, tuple) and len(node) == 2:
            x, y = node
        elif 'x' in G.nodes[node] and 'y' in G.nodes[node]:
            x = G.nodes[node]['x']
            y = G.nodes[node]['y']
        else:
            raise ValueError(f"Node {node} must be a (x, y) tuple or have 'x' and 'y' attributes")
        
        node_dict = {
            'id': idx,
            'n': G.nodes[node]['n'],
            'pos': [x, y]
        }
        # Include degree if available
        if 'degree' in G.nodes[node]:
            node_dict['degree'] = G.nodes[node]['degree']
        # Include closeness centrality (computed on raw graph)
        if node in closeness_centrality:
            node_dict['closeness_centrality'] = closeness_centrality[node]
        nodes.append(node_dict)
    
    # Create edge list
    edges = []
    for edge in G.edges():
        source_id = node_to_id[edge[0]]
        target_id = node_to_id[edge[1]]
        label = G.edges[edge]['n']  # 0, 1, or 2
        
        edges.append({
            'source': source_id,
            'target': target_id,
            'label': label
        })
    
    # Find crossing edges (conflicts) on raw graph
    conflicts = find_crossing_edges(G)
    
    # Store conflict information
    edge_conflicts = []
    for e1, e2 in conflicts:
        edge_conflicts.append({
            'edge1': {'source': node_to_id[e1[0]], 'target': node_to_id[e1[1]]},
            'edge2': {'source': node_to_id[e2[0]], 'target': node_to_id[e2[1]]}
        })
    
    return {
        'graph': {
            'nodes': nodes,
            'edges': edges,
            'edge_conflicts': edge_conflicts
        },
        'generation_params': generation_params,
        'split': split
    }


def create_dataset(
    output_dir: Path,
    num_puzzles: int,
    size: int = 8,
    difficulty: int = 0,
    islands_pct: int = 55,
    max_bridges: int = 2,
    expansion: Optional[int] = None,
    allow_loops: bool = False,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: Optional[int] = None
):
    """
    Create a dataset of Hashi puzzle graphs.
    
    Args:
        output_dir: Directory to save the dataset
        num_puzzles: Total number of puzzles to generate
        size: Puzzle size (width/height) (default: 8)
        difficulty: Puzzle difficulty (0-2) (default: 0)
        islands_pct: Percentage of islands (default: 55)
        max_bridges: Maximum number of bridges per connection (default: 2)
        expansion: Expansion percentage for island placement (default: None)
        allow_loops: Whether to allow loops in the puzzle (default: False)
        train_split: Fraction of puzzles for training (default: 0.7)
        val_split: Fraction of puzzles for validation (default: 0.15)
        test_split: Fraction of puzzles for testing (default: 0.15)
        seed: Random seed for reproducibility (default: None)
    
    Returns:
        None (creates files in output_dir)
    """
    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Create output directories
    raw_dir = output_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing puzzles
    next_puzzle_id, existing_splits = get_existing_puzzles(raw_dir)
    
    if next_puzzle_id > 0:
        print(f"Found {next_puzzle_id} existing puzzle(s) in dataset")
        print(f"  Existing splits - Train: {existing_splits['train']}, "
              f"Val: {existing_splits['val']}, Test: {existing_splits['test']}")
        print(f"  Starting new puzzles from ID: {next_puzzle_id}")
    
    # Generate puzzles and graphs
    print(f"Generating {num_puzzles} new puzzles...")
    puzzle_results = generate_puzzle_graph(
        count=num_puzzles,
        size=size,
        difficulty=difficulty,
        islands_pct=islands_pct,
        max_bridges=max_bridges,
        expansion=expansion,
        allow_loops=allow_loops
    )
    
    # Create splits
    indices = list(range(num_puzzles))
    random.shuffle(indices)
    
    train_end = int(num_puzzles * train_split)
    val_end = train_end + int(num_puzzles * val_split)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Save puzzles
    puzzle_id = next_puzzle_id
    new_splits = {'train': 0, 'val': 0, 'test': 0}
    
    for idx, (G, hashi_puzzle) in enumerate(puzzle_results):
        # Determine split
        if idx in train_indices:
            split = 'train'
        elif idx in val_indices:
            split = 'val'
        elif idx in test_indices:
            split = 'test'
        else:
            raise ValueError(f"Could not determine split for puzzle at index {idx}.")
        
        new_splits[split] += 1
        
        # Create puzzle data with all parameters (including defaults)
        generation_params = {
            'size': size,
            'difficulty': difficulty,
            'islands_pct': islands_pct,
            'max_bridges': max_bridges,
            'expansion': expansion,
            'allow_loops': allow_loops
        }
        
        puzzle_data = graph_to_dict(G, generation_params, split)
        
        # Save JSON file
        filename = f"puzzle_{puzzle_id:08d}.json"
        filepath = raw_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(puzzle_data, f, indent=2)
        
        puzzle_id += 1
    
    # Print summary
    total_train = existing_splits['train'] + new_splits['train']
    total_val = existing_splits['val'] + new_splits['val']
    total_test = existing_splits['test'] + new_splits['test']
    total_puzzles = next_puzzle_id + num_puzzles
    
    print("\nDataset updated successfully!")
    print(f"  New puzzles added: {num_puzzles}")
    print(f"    Training: {new_splits['train']}")
    print(f"    Validation: {new_splits['val']}")
    print(f"    Test: {new_splits['test']}")
    print(f"\n  Total puzzles in dataset: {total_puzzles}")
    print(f"    Training: {total_train}")
    print(f"    Validation: {total_val}")
    print(f"    Test: {total_test}")
    print(f"  Output directory: {output_dir}")
    print(f"  Raw files: {raw_dir}")


def main():
    """CLI entry point for create_data.py"""
    parser = argparse.ArgumentParser(
        description='Create a dataset of Hashi puzzle graphs for ML'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dataset',
        help='Output directory for dataset (default: dataset)'
    )
    parser.add_argument(
        '--num-puzzles',
        type=int,
        default=100,
        help='Total number of puzzles to generate (default: 100)'
    )
    parser.add_argument(
        '--size',
        type=int,
        nargs='+',
        default=[8],
        help='Puzzle size (width/height). Can be a list of sizes. (default: [8])'
    )
    parser.add_argument(
        '--difficulty',
        type=int,
        nargs='+',
        default=[0],
        choices=[0, 1, 2],
        help='Puzzle difficulty: 0=easy, 1=medium, 2=hard. Can be a list of difficulties. (default: [0])'
    )
    parser.add_argument(
        '--islands-pct',
        type=int,
        default=55,
        help='Percentage of islands (default: 55)'
    )
    parser.add_argument(
        '--max-bridges',
        type=int,
        default=2,
        help='Maximum number of bridges per connection (default: 2)'
    )
    parser.add_argument(
        '--expansion',
        type=int,
        default=None,
        help='Expansion percentage for island placement (default: None)'
    )
    parser.add_argument(
        '--allow-loops',
        action='store_true',
        help='Allow loops in the puzzle (default: False)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.7,
        help='Fraction of puzzles for training (default: 0.7)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Fraction of puzzles for validation (default: 0.15)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.15,
        help='Fraction of puzzles for testing (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Handle both single integer (from default) and list (from CLI)
    sizes = args.size if isinstance(args.size, list) else [args.size]
    difficulties = args.difficulty if isinstance(args.difficulty, list) else [args.difficulty]
    
    total_new_puzzles = 0
    
    # Generate puzzles for all combinations of sizes and difficulties
    for size in sizes:
        for difficulty in difficulties:
            print(f"\n--- Generating puzzles of size {size}x{size}, difficulty {difficulty} ---")
            # We need to re-read existing puzzles each time to get the correct next_id
            # However, create_dataset does this internally.
            # The main issue is that create_dataset expects to do everything.
            # We can just call it in a loop.
            
            create_dataset(
                output_dir=output_dir,
                num_puzzles=args.num_puzzles,
                size=size,
                difficulty=difficulty,
                islands_pct=args.islands_pct,
                max_bridges=args.max_bridges,
                expansion=args.expansion,
                allow_loops=args.allow_loops,
                train_split=args.train_split,
                val_split=args.val_split,
                test_split=args.test_split,
                seed=args.seed
            )
            total_new_puzzles += args.num_puzzles

    print("\n" + "="*40)
    print("Batch Generation Complete!")
    print(f"Total new puzzles generated: {total_new_puzzles}")
    print(f"  Across {len(sizes)} size(s) and {len(difficulties)} difficulty level(s)")
    print("="*40)


if __name__ == '__main__':
    main()

