#!/usr/bin/env python3
"""Find and render selected puzzles in the terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


def get_puzzle_size(data: dict[str, Any]) -> tuple[int, int]:
    """Calculate puzzle grid size from node positions."""
    positions = [node["pos"] for node in data["graph"]["nodes"]]
    max_x = max(p[0] for p in positions)
    max_y = max(p[1] for p in positions)
    return max_x + 1, max_y + 1


def render_puzzle(data: dict[str, Any], puzzle_file: Path) -> None:
    """Render a puzzle in the terminal."""
    nodes = data["graph"]["nodes"]
    edges = data["graph"]["edges"]
    width, height = get_puzzle_size(data)
    pos_to_node = {tuple(node["pos"]): node for node in nodes}

    print(f"\nPuzzle: {puzzle_file.name}")
    print(f"Size: {width}x{height}")
    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    print("=" * 80)

    for y in range(height):
        row = []
        for x in range(width):
            node = pos_to_node.get((x, y))
            row.append(f" {node['n']} " if node else " . ")
        print("".join(row))

    print("=" * 80)


def load_puzzle(puzzle_file: Path) -> dict[str, Any]:
    """Load one puzzle JSON file."""
    with puzzle_file.open() as f:
        return json.load(f)


def find_puzzles(dataset_path: Path, target_size: int, target_median: int) -> tuple[Any, Any]:
    """Find max-edge and median-edge puzzles for the target grid size."""
    puzzle_files = sorted(dataset_path.glob("puzzle_*.json"))
    max_edge_puzzle = None
    max_edge_count = 0
    median_edge_puzzle = None
    median_diff = float("inf")

    print(f"Scanning {len(puzzle_files)} puzzles...")
    for i, puzzle_file in enumerate(puzzle_files):
        if (i + 1) % 5000 == 0:
            print(f"  Scanned {i + 1}/{len(puzzle_files)} puzzles...")
        try:
            data = load_puzzle(puzzle_file)
            width, height = get_puzzle_size(data)
            if max(width, height) != target_size:
                continue
            num_nodes = len(data["graph"]["nodes"])
            num_edges = len(data["graph"]["edges"])
            if num_edges > max_edge_count:
                max_edge_count = num_edges
                max_edge_puzzle = (puzzle_file, data, num_nodes, num_edges)
            diff = abs(num_edges - target_median)
            if diff < median_diff:
                median_diff = diff
                median_edge_puzzle = (puzzle_file, data, num_nodes, num_edges)
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {puzzle_file}: {exc}", file=sys.stderr)
    return max_edge_puzzle, median_edge_puzzle


def select_failed_puzzles(
    diagnostic_csv: Path,
    preset: str,
    limit: int,
) -> list[str]:
    """Pick puzzle filenames that failed a given diagnostic preset."""
    df = pd.read_csv(diagnostic_csv)
    perfect_col = f"{preset}_perfect"
    wrong_col = f"{preset}_wrong_edges"
    if perfect_col not in df.columns:
        msg = f"Column not found: {perfect_col}"
        raise ValueError(msg)
    if wrong_col in df.columns:
        ordered = df[~df[perfect_col]].sort_values(wrong_col, ascending=False)
    else:
        ordered = df[~df[perfect_col]]
    return ordered["puzzle_file"].head(limit).tolist()


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Find and render selected Hashi puzzles.")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=str(Path(__file__).parent.parent / "dataset" / "raw"),
        help="Path to dataset raw directory",
    )
    parser.add_argument("--size", type=int, default=12, help="Target puzzle size")
    parser.add_argument(
        "--target-median",
        type=int,
        default=33,
        help="Target median edge count for second sample puzzle",
    )
    parser.add_argument(
        "--diagnostic-csv",
        type=Path,
        default=None,
        help="Optional diagnostic CSV to render failed puzzles",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="pure_noise",
        help="Noise preset used with --diagnostic-csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max failed puzzles to render when using --diagnostic-csv",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Searching puzzles in: {dataset_path}\n")

    if args.diagnostic_csv is not None:
        failures = select_failed_puzzles(args.diagnostic_csv, args.preset, args.limit)
        print(f"Rendering top {len(failures)} failures for preset '{args.preset}'")
        for filename in failures:
            puzzle_file = dataset_path / filename
            if not puzzle_file.exists():
                print(f"Missing puzzle file: {filename}", file=sys.stderr)
                continue
            render_puzzle(load_puzzle(puzzle_file), puzzle_file)
        return

    max_puzzle, median_puzzle = find_puzzles(
        dataset_path,
        target_size=args.size,
        target_median=args.target_median,
    )
    if max_puzzle:
        puzzle_file, data, _, _ = max_puzzle
        print("\n" + "=" * 80)
        print(f"PUZZLE WITH MAXIMUM EDGES ({args.size}x{args.size})")
        print("=" * 80)
        render_puzzle(data, puzzle_file)
    if median_puzzle:
        puzzle_file, data, _, _ = median_puzzle
        print("\n" + "=" * 80)
        print(f"PUZZLE WITH EDGES CLOSEST TO MEDIAN ({args.size}x{args.size})")
        print("=" * 80)
        render_puzzle(data, puzzle_file)


if __name__ == "__main__":
    main()
