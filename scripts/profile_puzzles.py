#!/usr/bin/env python3
"""
Profile puzzles in the raw dataset folder.
Analyzes distribution of node and edge counts by puzzle size.
"""

import json
from pathlib import Path
import sys
import pandas as pd

def get_puzzle_size(data):
    """Calculate puzzle grid size from node positions."""
    positions = [node['pos'] for node in data['graph']['nodes']]
    max_x = max(p[0] for p in positions)
    max_y = max(p[1] for p in positions)
    return max_x + 1, max_y + 1

def profile_puzzles(dataset_path):
    """Profile all puzzles in the dataset and return a DataFrame."""
    puzzle_data = []
    
    puzzle_files = sorted(Path(dataset_path).glob('puzzle_*.json'))
    total_puzzles = len(puzzle_files)
    
    print(f"Processing {total_puzzles} puzzles...")
    
    for i, puzzle_file in enumerate(puzzle_files):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{total_puzzles} puzzles...")
        
        try:
            with open(puzzle_file, 'r') as f:
                data = json.load(f)
            
            # Get puzzle size
            width, height = get_puzzle_size(data)
            size = max(width, height)
            
            # Count nodes and edges
            num_nodes = len(data['graph']['nodes'])
            num_edges = len(data['graph']['edges'])

            # Count nodes per capacity (1–8)
            capacity_counts = {f'cap_{c}': 0 for c in range(1, 9)}
            for node in data['graph']['nodes']:
                cap = node['n']
                if 1 <= cap <= 8:
                    capacity_counts[f'cap_{cap}'] += 1

            num_crossings = len(data['graph'].get('edge_conflicts', []))

            puzzle_data.append({
                'size': size,
                'nodes': num_nodes,
                'edges': num_edges,
                'crossings': num_crossings,
                **capacity_counts,
            })
            
        except Exception as e:
            print(f"Error processing {puzzle_file}: {e}", file=sys.stderr)
    
    return pd.DataFrame(puzzle_data)

def print_report(df):
    """Print a formatted report using pandas describe."""
    print("\n" + "="*80)
    print("PUZZLE PROFILE REPORT")
    print("="*80)
    print(f"\nTotal puzzles analyzed: {len(df):,}")
    print(f"Puzzle sizes found: {sorted(df['size'].unique().tolist())}")
    
    print("\n" + "="*80)
    print("NODE COUNT DISTRIBUTION BY PUZZLE SIZE")
    print("="*80)
    node_stats = df.groupby('size')['nodes'].describe()
    print(node_stats.to_string())
    
    print("\n" + "="*80)
    print("EDGE COUNT DISTRIBUTION BY PUZZLE SIZE")
    print("="*80)
    edge_stats = df.groupby('size')['edges'].describe()
    print(edge_stats.to_string())
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    summary = df.groupby('size').agg({
        'nodes': ['mean', 'std'],
        'edges': ['mean', 'std']
    }).round(2)
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary['edge_node_ratio'] = (df.groupby('size')['edges'].mean() / 
                                    df.groupby('size')['nodes'].mean()).round(2)
    summary['count'] = df.groupby('size').size()
    print(summary.to_string())
    cap_cols = [f'cap_{c}' for c in range(1, 9)]
    total = len(df)

    print("\n" + "="*80)
    print("CROSSING COUNT DISTRIBUTION BY PUZZLE SIZE")
    print("="*80)
    crossing_stats = df.groupby('size')['crossings'].describe().round(2)
    print(crossing_stats.to_string())


    print("\n" + "="*80)
    print("ISLAND CAPACITY PRESENCE (puzzles containing ≥1 island of each capacity)")
    print("="*80)
    presence = pd.DataFrame({
        'capacity': range(1, 9),
        'puzzles_with': [(df[col] > 0).sum() for col in cap_cols],
    })
    presence['pct_of_puzzles'] = (presence['puzzles_with'] / total * 100).round(2)
    print(presence.to_string(index=False))

    print("\n" + "="*80)
    print("ISLAND CAPACITY COUNT DISTRIBUTION (per-puzzle node counts, all puzzles)")
    print("="*80)
    cap_describe = df[cap_cols].describe().round(2)
    print(cap_describe.to_string())
    print("\n" + "="*80)

def main():
    # Default to dataset/raw in the current directory
    dataset_path = Path(__file__).parent.parent / 'dataset' / 'raw'
    
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
    
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Profiling puzzles in: {dataset_path}")
    
    df = profile_puzzles(dataset_path)
    print_report(df)

if __name__ == '__main__':
    main()
