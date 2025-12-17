"""
Utilities for constructing NetworkX graphs from Bridges/Hashi puzzles.

The graph represents islands as nodes and potential bridge connections as edges.
Edges have an 'n' attribute indicating the number of bridges (0 for potential, 1-2 for actual bridges).
"""
import json
import sys
import networkx as nx
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from .bridges_gen import generate_bridges
from .bridges_utils import convert_to_hashi_format

if sys.version_info >= (3, 10):
    from itertools import pairwise
else:
    from itertools import tee

    def pairwise(iterable):
        """Backport of itertools.pairwise for Python < 3.10."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for graph visualization functions. "
        "Install it with: pip install matplotlib"
    ) from exc


def puzzle_to_graph(puzzle: Dict) -> nx.Graph:
    """
    Convert a Bridges/Hashi puzzle to a NetworkX graph.
    
    The graph includes:
    - Nodes: Each island with 'n' attribute (island count)
    - Edges: All potential bridge connections with 'n' attribute:
      * 0: Potential connection (no bridge in solution)
      * 1: Single bridge in solution
      * 2: Double bridge in solution
    
    Args:
        puzzle: Puzzle dictionary with 'islands' and 'solution' keys
            - islands: List[{'x': int, 'y': int, 'count': int}]
            - solution: List[Tuple[int, int, int, int, str]] where str is '-' or '='
    
    Returns:
        NetworkX Graph with nodes and edges representing the puzzle
    """
    G = nx.Graph()
    x_dict = defaultdict(list)  # x -> [y1, y2, ...] for islands in same column
    y_dict = defaultdict(list)   # y -> [x1, x2, ...] for islands in same row
    
    # Add nodes and build coordinate dictionaries
    for island in puzzle['islands']:
        node = (island['x'], island['y'])
        G.add_node(node, n=island['count'])
        x_dict[island['x']].append(island['y'])
        y_dict[island['y']].append(island['x'])
    
    # Sort coordinate lists for consecutive edge creation
    for coords in x_dict.values():
        coords.sort()
    for coords in y_dict.values():
        coords.sort()
    
    # Add edges for all potential vertical connections (same x, consecutive y)
    for x, y_list in x_dict.items():
        for y1, y2 in pairwise(y_list):
            G.add_edge((x, y1), (x, y2), n=0)
    
    # Add edges for all potential horizontal connections (same y, consecutive x)
    for y, x_list in y_dict.items():
        for x1, x2 in pairwise(x_list):
            G.add_edge((x1, y), (x2, y), n=0)
    
    # Update edge attributes based on solution bridges
    for bridge in puzzle['solution']:
        x1, y1, x2, y2, bridge_type = bridge
        bridge_count = 1 if bridge_type == '-' else 2
        edge = ((x1, y1), (x2, y2))
        if G.has_edge(*edge):
            G.edges[edge]['n'] = bridge_count
    
    # Calculate and add node degree (based on all potential connections)
    for node in G.nodes():
        G.nodes[node]['degree'] = G.degree(node)
    
    return G


def generate_puzzle_graph(
    count: int = 1,
    size: int = 5,
    difficulty: int = 1,
    islands_pct: int = 55,
    max_bridges: int = 2,
    expansion: Optional[int] = None,
    allow_loops: bool = False
) -> List[Tuple[nx.Graph, Dict]]:
    """
    Generate Bridges puzzles and convert them to NetworkX graphs.
    
    This function wraps the full pipeline:
    1. Generate puzzles using bridges_gen
    2. Convert to hashi format
    3. Convert to NetworkX graphs
    
    Args:
        count: Number of puzzles to generate
        size: Puzzle size (width/height)
        difficulty: Puzzle difficulty (0-2)
        islands_pct: Percentage of islands (default 55)
        max_bridges: Maximum number of bridges (default: 2)
        expansion: Expansion percentage (default: None)
        allow_loops: Whether to allow loops (default: False)

    Returns:
        List of tuples, where each tuple contains (NetworkX Graph, hashi_puzzle_dict)
    """
    # Generate puzzles
    puzzles = generate_bridges(
        count=count, size=size, difficulty=difficulty, islands_pct=islands_pct,
        max_bridges=max_bridges, expansion=expansion, allow_loops=allow_loops
    )
    
    # Convert to hashi format and then to graphs
    results = []
    for puzzle in puzzles:
        hashi_puzzle = convert_to_hashi_format(puzzle)
        G = puzzle_to_graph(hashi_puzzle)
        results.append((G, hashi_puzzle))
    
    return results


def plot_graph_from_file(
    file_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    show_labels: bool = True,
    node_size: int = 1000,
    font_size: int = 10,
    edge_width: float = 2.0,
    edge_color: str = 'gray',
    node_color: str = 'lightblue',
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 100,
    save_format: str = 'png',
    save_dpi: int = 300,
    title: Optional[str] = None,
    layout: str = 'spring',
    pos: Optional[Dict[str, Tuple[float, float]]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> None:
    """
    Plots a NetworkX graph from a file.
    
    Args:
        file_path: Path to the JSON file containing the hashi puzzle.
        output_dir: Directory to save the plot. If None, plots to the current figure.
        show_labels: Whether to show node labels.
        node_size: Size of nodes.
        font_size: Font size for labels.
        edge_width: Width of edges.
        edge_color: Color of edges.
        node_color: Color of nodes.
        figsize: Size of the figure.
        dpi: DPI for saving.
        save_format: Format for saving (e.g., 'png', 'pdf').
        save_dpi: DPI for saving.
        title: Title for the plot.
        layout: NetworkX layout to use (e.g., 'spring', 'spectral', 'kamada_kawai').
        pos: Custom node positions.
        ax: Axes object to plot on.
        **kwargs: Additional arguments for nx.draw.
    """
    # Load the puzzle from the file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the graph data from the JSON structure
    graph_data = data['graph']
    
    # Reconstruct NetworkX graph directly from JSON
    G = nx.Graph()
    
    # Build node_id -> coordinates mapping
    node_map = {}
    for node in graph_data['nodes']:
        node_id = node['id']
        coords = tuple(node['pos'])
        node_map[node_id] = coords
        G.add_node(coords, n=node['n'])
    
    # Add edges with labels
    for edge in graph_data['edges']:
        source_coords = node_map[edge['source']]
        target_coords = node_map[edge['target']]
        G.add_edge(source_coords, target_coords, n=edge['label'])

    # Set layout
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif pos is None:
        pos = nx.spring_layout(G) # Default to spring if no custom pos

    # Plot
    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()

    nx.draw(
        G,
        pos=pos,
        with_labels=show_labels,
        node_size=node_size,
        font_size=font_size,
        edge_color=edge_color,
        node_color=node_color,
        ax=ax,
        **kwargs
    )

    # Set title
    if title:
        ax.set_title(title)

    # Save
    if output_dir:
        output_path = Path(output_dir) / (Path(file_path).stem + '.' + save_format)
        plt.savefig(output_path, dpi=save_dpi)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
