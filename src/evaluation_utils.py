"""
Evaluation utilities for Hashi graph model analysis.

This module provides functions for:
- Error classification and analysis
- Visualization helpers
- Statistical analysis
- Model interpretability tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
from pathlib import Path


def check_puzzle_connectivity(puzzle_data: Dict, predicted_edges: List[Tuple]) -> bool:
    """
    Check if predicted edges form a fully connected graph.

    Args:
        puzzle_data: Raw puzzle data from JSON
        predicted_edges: List of (node1_id, node2_id, bridge_count) tuples

    Returns:
        True if all islands are connected, False otherwise
    """
    # Build graph from predicted edges
    G = nx.Graph()

    # Add all islands as nodes
    for node in puzzle_data['graph']['nodes']:
        G.add_node(node['id'])

    # Add edges for predicted bridges
    for n1_id, n2_id, bridges in predicted_edges:
        if bridges > 0:
            G.add_edge(n1_id, n2_id)

    # Check connectivity (excluding meta nodes if present)
    puzzle_nodes = [n['id'] for n in puzzle_data['graph']['nodes'] if n['n'] <= 8]
    subgraph = G.subgraph(puzzle_nodes)

    return nx.is_connected(subgraph) if len(subgraph.nodes) > 1 else True


def check_capacity_constraints(puzzle_data: Dict, predicted_edges: List[Tuple]) -> Dict[str, Any]:
    """
    Check capacity constraints for predicted edges.

    Args:
        puzzle_data: Raw puzzle data from JSON
        predicted_edges: List of (node1_id, node2_id, bridge_count) tuples

    Returns:
        Dictionary with capacity violation analysis
    """
    # Calculate usage per node
    usage = defaultdict(int)
    for n1_id, n2_id, bridges in predicted_edges:
        usage[n1_id] += bridges
        usage[n2_id] += bridges

    # Check violations
    violations = []
    total_capacity = 0
    total_used = 0

    for node in puzzle_data['graph']['nodes']:
        if node['n'] <= 8:  # Only check puzzle nodes
            node_id = node['id']
            capacity = node['n']
            used = usage.get(node_id, 0)

            total_capacity += capacity
            total_used += used

            if used > capacity:
                violations.append({
                    'node_id': node_id,
                    'capacity': capacity,
                    'used': used,
                    'excess': used - capacity
                })

    return {
        'violations': violations,
        'total_capacity': total_capacity,
        'total_used': total_used,
        'capacity_utilization': total_used / total_capacity if total_capacity > 0 else 0,
        'has_violations': len(violations) > 0
    }


def classify_puzzle_error_types(puzzle_data: Dict, predicted_edges: List[Tuple],
                               actual_edges: List[Tuple]) -> Dict[str, Any]:
    """
    Classify error types for a single puzzle.

    Args:
        puzzle_data: Raw puzzle data from JSON
        predicted_edges: List of (node1_id, node2_id, bridge_count) tuples
        actual_edges: List of (node1_id, node2_id, bridge_count) tuples

    Returns:
        Dictionary with error classification
    """

    # Build lookup dictionaries
    pred_lookup = {(n1, n2): bridges for n1, n2, bridges in predicted_edges}
    actual_lookup = {(n1, n2): bridges for n1, n2, bridges in actual_edges}

    # Find all edges present in either prediction or actual
    all_edges = set(pred_lookup.keys()) | set(actual_lookup.keys())

    error_counts = {
        'bridge_count_errors': 0,  # Wrong number of bridges
        'missing_bridges': 0,      # Predicted 0 but should have bridges
        'extra_bridges': 0,        # Predicted bridges but should be 0
        'total_errors': 0
    }

    for edge in all_edges:
        pred = pred_lookup.get(edge, 0)
        actual = actual_lookup.get(edge, 0)

        if pred != actual:
            error_counts['total_errors'] += 1

            if actual == 0 and pred > 0:
                error_counts['extra_bridges'] += 1
            elif actual > 0 and pred == 0:
                error_counts['missing_bridges'] += 1
            elif actual > 0 and pred > 0:
                error_counts['bridge_count_errors'] += 1

    # Check connectivity and capacity
    connectivity_ok = check_puzzle_connectivity(puzzle_data, predicted_edges)
    capacity_analysis = check_capacity_constraints(puzzle_data, predicted_edges)

    return {
        'error_counts': error_counts,
        'connectivity_ok': connectivity_ok,
        'capacity_analysis': capacity_analysis,
        'error_types': []
    }


def plot_error_heatmap(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create a heatmap showing error patterns across puzzle characteristics.
    """
    # Aggregate errors by puzzle size and difficulty
    heatmap_data = results_df.groupby(['puzzle_size', 'puzzle_difficulty']).agg({
        'error': ['count', lambda x: (x > 0).mean(), 'mean']
    }).round(3)

    heatmap_data.columns = ['total_edges', 'error_rate', 'avg_error']
    heatmap_data = heatmap_data.reset_index()

    # Create pivot tables
    error_rate_pivot = heatmap_data.pivot(
        index='puzzle_size',
        columns='puzzle_difficulty',
        values='error_rate'
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Error rate heatmap
    sns.heatmap(error_rate_pivot, annot=True, fmt='.3f', cmap='Reds', ax=ax1)
    ax1.set_title('Error Rate by Puzzle Size and Difficulty')
    ax1.set_xlabel('Difficulty')
    ax1.set_ylabel('Size')

    # Average error heatmap
    avg_error_pivot = heatmap_data.pivot(
        index='puzzle_size',
        columns='puzzle_difficulty',
        values='avg_error'
    )

    sns.heatmap(avg_error_pivot, annot=True, fmt='.3f', cmap='Oranges', ax=ax2)
    ax2.set_title('Average Error Magnitude by Puzzle Size and Difficulty')
    ax2.set_xlabel('Difficulty')
    ax2.set_ylabel('Size')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, (ax1, ax2)


def plot_capacity_error_analysis(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Analyze and plot errors by node capacity.
    """
    # Calculate error rates by capacity
    capacity_stats = []

    for cap in sorted(results_df['node1_capacity'].unique()):
        # Edges involving nodes of this capacity
        cap_edges = results_df[
            (results_df['node1_capacity'] == cap) |
            (results_df['node2_capacity'] == cap)
        ]

        # Edges where this capacity node has an error
        cap_errors = cap_edges[cap_edges['error'] > 0]

        capacity_stats.append({
            'capacity': cap,
            'total_edges': len(cap_edges),
            'error_count': len(cap_errors),
            'error_rate': len(cap_errors) / len(cap_edges) if len(cap_edges) > 0 else 0,
            'avg_error_magnitude': cap_errors['error'].mean() if len(cap_errors) > 0 else 0
        })

    cap_df = pd.DataFrame(capacity_stats)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Error rate by capacity
    axes[0, 0].bar(cap_df['capacity'], cap_df['error_rate'], color='lightcoral', alpha=0.7)
    axes[0, 0].set_xlabel('Node Capacity')
    axes[0, 0].set_ylabel('Error Rate')
    axes[0, 0].set_title('Error Rate by Node Capacity')
    axes[0, 0].set_xticks(cap_df['capacity'])

    # Average error magnitude by capacity
    axes[0, 1].bar(cap_df['capacity'], cap_df['avg_error_magnitude'], color='orange', alpha=0.7)
    axes[0, 1].set_xlabel('Node Capacity')
    axes[0, 1].set_ylabel('Average Error Magnitude')
    axes[0, 1].set_title('Average Error Magnitude by Node Capacity')
    axes[0, 1].set_xticks(cap_df['capacity'])

    # Total edges by capacity
    axes[1, 0].bar(cap_df['capacity'], cap_df['total_edges'], color='lightblue', alpha=0.7)
    axes[1, 0].set_xlabel('Node Capacity')
    axes[1, 0].set_ylabel('Total Edges')
    axes[1, 0].set_title('Edge Count by Node Capacity')
    axes[1, 0].set_xticks(cap_df['capacity'])

    # Error count by capacity
    axes[1, 1].bar(cap_df['capacity'], cap_df['error_count'], color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Node Capacity')
    axes[1, 1].set_ylabel('Error Count')
    axes[1, 1].set_title('Error Count by Node Capacity')
    axes[1, 1].set_xticks(cap_df['capacity'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def analyze_error_patterns_by_position(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Analyze how errors vary by node positions in the grid.
    """
    # Calculate error rates by grid position (simplified to manhattan distance from center)
    position_stats = []

    for _, row in results_df.iterrows():
        # Calculate distances from grid center for both nodes
        size = row['puzzle_size']
        center = size / 2.0

        pos1 = np.array(row['node1_pos'])
        pos2 = np.array(row['node2_pos'])

        dist1 = np.linalg.norm(pos1 - center)
        dist2 = np.linalg.norm(pos2 - center)
        avg_dist = (dist1 + dist2) / 2

        position_stats.append({
            'edge_idx': row['edge_idx'],
            'avg_distance_from_center': avg_dist,
            'error': row['error'],
            'has_error': row['error'] > 0
        })

    pos_df = pd.DataFrame(position_stats)

    # Bin by distance
    pos_df['distance_bin'] = pd.cut(pos_df['avg_distance_from_center'],
                                   bins=5, labels=['Center', 'Near', 'Middle', 'Far', 'Edge'])

    # Calculate stats by distance bin
    bin_stats = pos_df.groupby('distance_bin').agg({
        'has_error': 'mean',
        'error': 'mean',
        'edge_idx': 'count'
    }).round(3)

    bin_stats.columns = ['error_rate', 'avg_error', 'count']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Error rate by position
    bin_stats['error_rate'].plot(kind='bar', ax=ax1, color='lightgreen', alpha=0.7)
    ax1.set_xlabel('Distance from Center')
    ax1.set_ylabel('Error Rate')
    ax1.set_title('Error Rate by Grid Position')
    ax1.tick_params(axis='x', rotation=45)

    # Average error by position
    bin_stats['avg_error'].plot(kind='bar', ax=ax2, color='orange', alpha=0.7)
    ax2.set_xlabel('Distance from Center')
    ax2.set_ylabel('Average Error Magnitude')
    ax2.set_title('Average Error by Grid Position')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, (ax1, ax2), bin_stats


def create_model_comparison_summary(results_df: pd.DataFrame,
                                   model_name: str,
                                   additional_metrics: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create a comprehensive summary of model performance.

    Args:
        results_df: Results dataframe from evaluation
        model_name: Name/identifier for the model
        additional_metrics: Optional additional metrics to include

    Returns:
        Dictionary with comprehensive performance summary
    """

    # Basic metrics
    total_edges = len(results_df)
    total_errors = (results_df['error'] > 0).sum()
    accuracy = (results_df['error'] == 0).mean()

    # Perfect puzzle metrics
    perfect_puzzles = results_df.groupby('puzzle_idx')['is_perfect'].first()
    perfect_rate = perfect_puzzles.mean()

    # Error distribution
    error_dist = results_df['error'].value_counts().to_dict()

    # Performance by puzzle characteristics
    size_performance = results_df.groupby('puzzle_size').agg({
        'error': lambda x: (x == 0).mean()
    }).round(4).to_dict()['error']

    difficulty_performance = results_df.groupby('puzzle_difficulty').agg({
        'error': lambda x: (x == 0).mean()
    }).round(4).to_dict()['error']

    # Capacity analysis
    capacity_performance = {}
    for cap in sorted(results_df['node1_capacity'].unique()):
        cap_edges = results_df[
            (results_df['node1_capacity'] == cap) |
            (results_df['node2_capacity'] == cap)
        ]
        capacity_performance[int(cap)] = (cap_edges['error'] == 0).mean()

    summary = {
        'model_name': model_name,
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_info': {
            'total_puzzles': len(results_df['puzzle_idx'].unique()),
            'total_edges': total_edges,
            'puzzle_sizes': sorted(results_df['puzzle_size'].unique()),
            'difficulties': sorted(results_df['puzzle_difficulty'].unique())
        },
        'overall_metrics': {
            'edge_accuracy': accuracy,
            'perfect_puzzle_rate': perfect_rate,
            'total_errors': total_errors,
            'error_rate': total_errors / total_edges,
            'mean_error_magnitude': results_df['error'].mean()
        },
        'error_distribution': error_dist,
        'performance_by_size': size_performance,
        'performance_by_difficulty': difficulty_performance,
        'performance_by_capacity': capacity_performance
    }

    if additional_metrics:
        summary['additional_metrics'] = additional_metrics

    return summary



