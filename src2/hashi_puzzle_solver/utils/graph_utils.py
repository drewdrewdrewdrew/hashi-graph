"""
Utilities for constructing NetworkX graphs from Bridges/Hashi puzzles.

The graph represents islands as nodes and potential bridge connections as edges.
Edges have an 'n' attribute indicating the number of bridges
(0 for potential, 1-2 for actual bridges).
"""

from collections import defaultdict
from itertools import pairwise
import json
from pathlib import Path

import networkx as nx

from .bridges_gen import generate_bridges
from .bridges_utils import convert_to_hashi_format

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    msg = (
        "matplotlib is required for graph visualization functions. "
        "Install it with: pip install matplotlib"
    )
    raise ImportError(msg) from exc


def _add_nodes_and_build_coords(puzzle: dict, g: nx.Graph) -> tuple[dict, dict]:
    """Add nodes to graph and build coordinate dictionaries."""
    x_dict = defaultdict(list)  # x -> [y1, y2, ...] for islands in same column
    y_dict = defaultdict(list)  # y -> [x1, x2, ...] for islands in same row

    for island in puzzle["islands"]:
        node = (island["x"], island["y"])
        g.add_node(node, n=island["count"])
        x_dict[island["x"]].append(island["y"])
        y_dict[island["y"]].append(island["x"])

    # Sort coordinate lists for consecutive edge creation
    for coords in x_dict.values():
        coords.sort()
    for coords in y_dict.values():
        coords.sort()

    return x_dict, y_dict


def _add_potential_edges(g: nx.Graph, x_dict: dict, y_dict: dict) -> None:
    """Add edges for all potential bridge connections."""
    # Add edges for all potential vertical connections (same x, consecutive y)
    for x, y_list in x_dict.items():
        for y1, y2 in pairwise(y_list):
            g.add_edge((x, y1), (x, y2), n=0)

    # Add edges for all potential horizontal connections (same y, consecutive x)
    for y, x_list in y_dict.items():
        for x1, x2 in pairwise(x_list):
            g.add_edge((x1, y), (x2, y), n=0)


def _update_bridge_attributes(g: nx.Graph, puzzle: dict) -> None:
    """Update edge attributes based on solution bridges."""
    for bridge in puzzle["solution"]:
        x1, y1, x2, y2, bridge_type = bridge
        bridge_count = 1 if bridge_type == "-" else 2
        edge = ((x1, y1), (x2, y2))
        if g.has_edge(*edge):
            g.edges[edge]["n"] = bridge_count


def puzzle_to_graph(puzzle: dict) -> nx.Graph:
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

    Returns
    -------
        NetworkX Graph with nodes and edges representing the puzzle
    """
    g = nx.Graph()

    x_dict, y_dict = _add_nodes_and_build_coords(puzzle, g)
    _add_potential_edges(g, x_dict, y_dict)
    _update_bridge_attributes(g, puzzle)

    # Calculate and add node degree (based on all potential connections)
    for node in g.nodes():
        g.nodes[node]["degree"] = g.degree(node)

    return g


def generate_puzzle_graph(
    count: int = 1,
    size: int = 5,
    difficulty: int = 1,
    islands_pct: int = 55,
    max_bridges: int = 2,
    expansion: int | None = None,
    allow_loops: bool = False,
) -> list[tuple[nx.Graph, dict]]:
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

    Returns
    -------
        List of tuples, where each tuple contains (NetworkX Graph, hashi_puzzle_dict)
    """
    # Generate puzzles
    puzzles = generate_bridges(
        count=count,
        size=size,
        difficulty=difficulty,
        islands_pct=islands_pct,
        max_bridges=max_bridges,
        expansion=expansion,
        allow_loops=allow_loops,
    )

    # Convert to hashi format and then to graphs
    results = []
    for puzzle in puzzles:
        hashi_puzzle = convert_to_hashi_format(puzzle)
        g = puzzle_to_graph(hashi_puzzle)
        results.append((g, hashi_puzzle))

    return results


def plot_graph_from_file(
    file_path: str | Path,
    output_dir: str | Path | None = None,
    show_labels: bool = True,
    node_size: int = 1000,
    font_size: int = 10,
    _edge_width: float = 2.0,
    edge_color: str = "gray",
    node_color: str = "lightblue",
    figsize: tuple[int, int] = (10, 10),
    dpi: int = 100,
    save_format: str = "png",
    save_dpi: int = 300,
    title: str | None = None,
    layout: str = "spring",
    pos: dict[str, tuple[float, float]] | None = None,
    ax: plt.Axes | None = None,
    **kwargs: dict,
) -> None:
    """
    Plot a NetworkX graph from a file.

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
    with Path(file_path).open() as f:
        data = json.load(f)

    # Extract the graph data from the JSON structure
    graph_data = data["graph"]

    # Reconstruct NetworkX graph directly from JSON
    g = nx.Graph()

    # Build node_id -> coordinates mapping
    node_map = {}
    for node in graph_data["nodes"]:
        node_id = node["id"]
        coords = tuple(node["pos"])
        node_map[node_id] = coords
        g.add_node(coords, n=node["n"])

    # Add edges with labels
    for edge in graph_data["edges"]:
        source_coords = node_map[edge["source"]]
        target_coords = node_map[edge["target"]]
        g.add_edge(source_coords, target_coords, n=edge["label"])

    # Set layout
    if layout == "spring":
        pos = nx.spring_layout(g)
    elif layout == "spectral":
        pos = nx.spectral_layout(g)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(g)
    elif pos is None:
        pos = nx.spring_layout(g)  # Default to spring if no custom pos

    # Plot
    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()

    nx.draw(
        g,
        pos=pos,
        with_labels=show_labels,
        node_size=node_size,
        font_size=font_size,
        edge_color=edge_color,
        node_color=node_color,
        ax=ax,
        **kwargs,
    )

    # Set title
    if title:
        ax.set_title(title)

    # Save
    if output_dir:
        output_path = Path(output_dir) / (Path(file_path).stem + "." + save_format)
        plt.savefig(output_path, dpi=save_dpi)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
