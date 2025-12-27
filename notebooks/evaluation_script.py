"""
Hashi Graph Model Evaluation Script

This script provides comprehensive evaluation of trained Hashi graph models.
Run this instead of the notebook if you have issues with the notebook format.

Usage:
    python notebooks/evaluation_script.py
"""

import sys
import os
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('../src')

# Import project modules
from data import HashiDataset, custom_collate_with_conflicts
from models import GCNEdgeClassifier, GATEdgeClassifier, GINEEdgeClassifier, TransformerEdgeClassifier
from bridges_gen import generate_bridges
from bridges_utils import convert_to_hashi_format
from graph_utils import puzzle_to_graph
from train_utils import calculate_batch_perfect_puzzles, get_edge_batch_indices
from evaluation_utils import (classify_puzzle_error_types, plot_error_heatmap,
                              plot_capacity_error_analysis, analyze_error_patterns_by_position,
                              create_model_comparison_summary)

# Set style
plt.style.use('default')
sns.set_palette("husl")

def main():
    # Configuration - MODIFY THESE PATHS
    MODEL_PATH = "../models/model_20241220_120000/best_model.pt"  # Path to saved model
    CONFIG_PATH = "../configs/base_config.yaml"  # Path to config used for training
    DATA_ROOT = "../data"  # Path to dataset root
    SPLIT = "val"  # 'val' or 'test'
    LIMIT = 10  # Limit number of puzzles for testing (set to None for all)

    # Analysis settings
    SAVE_PLOTS = True
    PLOT_DIR = "../plots"
    if SAVE_PLOTS:
        Path(PLOT_DIR).mkdir(exist_ok=True)

    print("=== Hashi Graph Model Evaluation ===\n")

    # Load configuration
    print("Loading configuration...")
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config from {CONFIG_PATH}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available()
                         else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("\nLoading model...")
    try:
        model = load_model(config, device, MODEL_PATH)
        print(f"✓ Loaded {config['model']['type']} model")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset, dataloader = load_dataset(config, DATA_ROOT, SPLIT, LIMIT)
        print(f"✓ Loaded {len(dataset)} puzzles for evaluation")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Run inference
    print("\nRunning inference...")
    try:
        results_df = run_inference(model, dataloader, device)
        print(f"✓ Collected results for {len(results_df)} edges across {results_df['puzzle_idx'].nunique()} puzzles")
    except Exception as e:
        print(f"✗ Failed to run inference: {e}")
        return

    # Basic analysis
    print("\n=== Basic Analysis ===")
    perfect_puzzles = results_df.groupby('puzzle_idx')['is_perfect'].first().sum()
    total_puzzles = results_df['puzzle_idx'].nunique()
    accuracy = (results_df['error'] == 0).mean()

    print(".1f")
    print(".1f")
    print(".4f")

    # Error classification
    print("\n=== Error Classification ===")
    try:
        puzzle_errors_df = classify_puzzle_errors(results_df)
        error_summary = puzzle_errors_df['error_types'].explode().value_counts()
        print("Error types found:")
        for error_type, count in error_summary.items():
            print(f"  {error_type}: {count}")
    except Exception as e:
        print(f"✗ Failed to classify errors: {e}")

    # Generate plots
    if SAVE_PLOTS:
        print(f"\n=== Generating Plots ===")
        try:
            plot_error_patterns(results_df, PLOT_DIR)
            print(f"✓ Saved error pattern plots to {PLOT_DIR}")
        except Exception as e:
            print(f"✗ Failed to generate plots: {e}")

    # Export results
    if SAVE_PLOTS:
        results_df.to_csv(f"{PLOT_DIR}/evaluation_results.csv", index=False)
        print(f"✓ Exported results to {PLOT_DIR}/evaluation_results.csv")

    print("\n=== Evaluation Complete ===")
    print("Check the plots directory for visualizations!")

def load_model(config: Dict[str, Any], device: torch.device, model_path: str) -> nn.Module:
    """Load and initialize the model from config."""
    model_config = config['model']
    model_type = model_config.get('type', 'gcn').lower()

    # Get model feature settings
    use_capacity = model_config.get('use_capacity', True)
    use_structural_degree = model_config.get('use_structural_degree', True)
    use_structural_degree_nsew = model_config.get('use_structural_degree_nsew', False)
    use_unused_capacity = model_config.get('use_unused_capacity', True)
    use_conflict_status = model_config.get('use_conflict_status', True)
    use_global_meta_node = model_config.get('use_global_meta_node', False)
    use_row_col_meta = model_config.get('use_row_col_meta', False)
    use_closeness_centrality = model_config.get('use_closeness_centrality', False)

    # Determine edge dimension
    edge_dim = 3  # base: [inv_dx, inv_dy, is_meta]
    if model_config.get('use_conflict_edges', False):
        edge_dim += 1
    if model_config.get('use_meta_mesh', False):
        edge_dim += 1
    if model_config.get('use_meta_row_col_edges', False):
        edge_dim += 1
    if model_config.get('use_edge_labels_as_features', False):
        edge_dim += 2

    # Initialize model
    if model_type == 'gcn':
        model = GCNEdgeClassifier(
            node_embedding_dim=model_config['node_embedding_dim'],
            hidden_channels=model_config['hidden_channels'],
            num_layers=model_config['num_layers'],
            dropout=model_config.get('dropout', 0.25),
            use_capacity=use_capacity,
            use_structural_degree=use_structural_degree,
            use_structural_degree_nsew=use_structural_degree_nsew,
            use_unused_capacity=use_unused_capacity,
            use_conflict_status=use_conflict_status,
            use_meta_node=use_global_meta_node,
            use_closeness=use_closeness_centrality
        )
    elif model_type == 'gat':
        model = GATEdgeClassifier(
            node_embedding_dim=model_config['node_embedding_dim'],
            hidden_channels=model_config['hidden_channels'],
            num_layers=model_config['num_layers'],
            heads=model_config.get('heads', 8),
            dropout=model_config.get('dropout', 0.25),
            use_capacity=use_capacity,
            use_structural_degree=use_structural_degree,
            use_structural_degree_nsew=use_structural_degree_nsew,
            use_unused_capacity=use_unused_capacity,
            use_conflict_status=use_conflict_status,
            use_meta_node=use_global_meta_node,
            use_row_col_meta=use_row_col_meta,
            edge_dim=edge_dim,
            use_closeness=use_closeness_centrality
        )
    elif model_type == 'gine':
        model = GINEEdgeClassifier(
            node_embedding_dim=model_config['node_embedding_dim'],
            hidden_channels=model_config['hidden_channels'],
            num_layers=model_config['num_layers'],
            dropout=model_config.get('dropout', 0.25),
            use_capacity=use_capacity,
            use_structural_degree=use_structural_degree,
            use_structural_degree_nsew=use_structural_degree_nsew,
            use_unused_capacity=use_unused_capacity,
            use_conflict_status=use_conflict_status,
            use_meta_node=use_global_meta_node,
            use_row_col_meta=use_row_col_meta,
            edge_dim=edge_dim,
            use_closeness=use_closeness_centrality
        )
    elif model_type == 'transformer':
        model = TransformerEdgeClassifier(
            node_embedding_dim=model_config['node_embedding_dim'],
            hidden_channels=model_config['hidden_channels'],
            num_layers=model_config['num_layers'],
            heads=model_config.get('heads', 4),
            dropout=model_config.get('dropout', 0.25),
            use_capacity=use_capacity,
            use_structural_degree=use_structural_degree,
            use_structural_degree_nsew=use_structural_degree_nsew,
            use_unused_capacity=use_unused_capacity,
            use_conflict_status=use_conflict_status,
            use_meta_node=use_global_meta_node,
            use_row_col_meta=use_row_col_meta,
            edge_dim=edge_dim,
            use_closeness=use_closeness_centrality,
            use_verification_head=model_config.get('use_verification_head', False),
            verifier_use_puzzle_nodes=model_config.get('verifier_use_puzzle_nodes', False),
            verifier_use_row_col_meta_nodes=model_config.get('verifier_use_row_col_meta_nodes', False),
            edge_concat_global_meta=model_config.get('edge_concat_global_meta', False)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

def load_dataset(config, data_root, split, limit):
    """Load dataset with proper parameters."""
    data_config = config['data']
    model_config = config['model']

    # Get dataset parameters from config
    dataset_params = {
        'use_degree': model_config.get('use_degree', False),
        'use_meta_node': model_config.get('use_global_meta_node', False),
        'use_row_col_meta': model_config.get('use_row_col_meta', False),
        'use_meta_mesh': model_config.get('use_meta_mesh', False),
        'use_meta_row_col_edges': model_config.get('use_meta_row_col_edges', False),
        'use_distance': model_config.get('use_distance', True),
        'use_edge_labels_as_features': model_config.get('use_edge_labels_as_features', False),
        'use_closeness_centrality': model_config.get('use_closeness_centrality', False),
        'use_conflict_edges': model_config.get('use_conflict_edges', False),
        'use_capacity': model_config.get('use_capacity', True),
        'use_structural_degree': model_config.get('use_structural_degree', True),
        'use_structural_degree_nsew': model_config.get('use_structural_degree_nsew', False),
        'use_unused_capacity': model_config.get('use_unused_capacity', True),
        'use_conflict_status': model_config.get('use_conflict_status', True),
    }

    dataset = HashiDataset(
        root=data_root,
        split=split,
        size=data_config.get('size'),
        difficulty=data_config.get('difficulty'),
        limit=limit,
        **dataset_params
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one puzzle at a time for detailed analysis
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_with_conflicts
    )

    return dataset, dataloader

def run_inference(model: nn.Module, dataloader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Run inference on the dataset and collect detailed results."""
    results = []

    model.eval()
    with torch.no_grad():
        for puzzle_idx, data in enumerate(tqdm(dataloader, desc="Running inference")):
            data = data.to(device)

            # Forward pass
            logits = model(
                data.x, data.edge_index,
                edge_attr=getattr(data, 'edge_attr', None),
                batch=getattr(data, 'batch', None)
            )

            # Get predictions for original edges only
            logits_original = logits[data.edge_mask]
            predictions = logits_original.argmax(dim=-1)
            probabilities = torch.softmax(logits_original, dim=-1)
            pred_probs = probabilities.gather(1, predictions.unsqueeze(-1)).squeeze(-1)

            # Ground truth
            actual = data.y

            # Calculate if puzzle is perfect
            is_perfect = (predictions == actual).all().item()

            # Get node information
            edge_index = data.edge_index[:, data.edge_mask]
            num_puzzle_nodes = (data.x[:, 0] <= 8).sum().item()  # Puzzle nodes have capacity <= 8

            # Extract puzzle metadata (from the raw file)
            raw_path = dataloader.dataset.raw_file_names[puzzle_idx]
            with open(Path(dataloader.dataset.raw_dir) / raw_path, 'r') as f:
                puzzle_data = json.load(f)

            puzzle_size = puzzle_data['generation_params']['size']
            puzzle_difficulty = puzzle_data['generation_params']['difficulty']

            # Node positions and capacities
            node_pos = {}
            node_capacity = {}
            for node in puzzle_data['graph']['nodes']:
                node_pos[node['id']] = node['pos']
                node_capacity[node['id']] = node['n']

            # Process each edge
            for edge_idx in range(len(actual)):
                n1_idx = edge_index[0, edge_idx].item()
                n2_idx = edge_index[1, edge_idx].item()

                # Only process puzzle node edges (not meta edges)
                if n1_idx >= num_puzzle_nodes or n2_idx >= num_puzzle_nodes:
                    continue

                # Get node IDs from the raw data
                edge_info = puzzle_data['graph']['edges'][edge_idx]
                n1_id = edge_info['source']
                n2_id = edge_info['target']

                result = {
                    'puzzle_idx': puzzle_idx,
                    'edge_idx': edge_idx,
                    'node1_id': n1_id,
                    'node2_id': n2_id,
                    'node1_pos': node_pos[n1_id],
                    'node2_pos': node_pos[n2_id],
                    'node1_capacity': node_capacity[n1_id],
                    'node2_capacity': node_capacity[n2_id],
                    'actual_bridge': actual[edge_idx].item(),
                    'pred_bridge': predictions[edge_idx].item(),
                    'pred_prob': pred_probs[edge_idx].item(),
                    'error': abs(actual[edge_idx].item() - predictions[edge_idx].item()),
                    'is_perfect': is_perfect,
                    'puzzle_size': puzzle_size,
                    'puzzle_difficulty': puzzle_difficulty
                }
                results.append(result)

    return pd.DataFrame(results)

def classify_puzzle_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Classify non-perfect puzzles into error categories."""
    puzzle_errors = []

    for puzzle_idx in df['puzzle_idx'].unique():
        puzzle_data = df[df['puzzle_idx'] == puzzle_idx]

        # Load raw puzzle data for detailed analysis
        raw_path = df.dataset.raw_file_names[puzzle_idx] if hasattr(df, 'dataset') else f"puzzle_{puzzle_idx:06d}.json"
        # Simplified version without full path resolution for now
        error_types = []
        if not puzzle_data['is_perfect'].iloc[0]:
            # Basic error classification
            bridge_errors = puzzle_data[
                (puzzle_data['actual_bridge'] > 0) &
                (puzzle_data['pred_bridge'] != puzzle_data['actual_bridge'])
            ]
            if len(bridge_errors) > 0:
                error_types.append('bridge_count_error')

            # Check for capacity violations (simplified)
            capacity_usage = defaultdict(int)
            for _, row in puzzle_data.iterrows():
                pred_bridges = row['pred_bridge']
                capacity_usage[row['node1_id']] += pred_bridges
                capacity_usage[row['node2_id']] += pred_bridges

            capacity_violations = False
            for node_id, usage in capacity_usage.items():
                capacity = puzzle_data[puzzle_data['node1_id'] == node_id]['node1_capacity'].iloc[0]
                if usage > capacity:
                    capacity_violations = True
                    break

            if capacity_violations:
                error_types.append('capacity_error')

        puzzle_errors.append({
            'puzzle_idx': puzzle_idx,
            'error_types': error_types,
            'num_errors': len(puzzle_data[puzzle_data['error'] > 0]),
            'error_rate': (puzzle_data['error'] > 0).mean(),
            'puzzle_size': puzzle_data['puzzle_size'].iloc[0],
            'puzzle_difficulty': puzzle_data['puzzle_difficulty'].iloc[0]
        })

    return pd.DataFrame(puzzle_errors)

def plot_error_patterns(df: pd.DataFrame, save_dir: Optional[str] = None):
    """Create basic error pattern visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Error Pattern Analysis', fontsize=14, fontweight='bold')

    # Error distribution
    error_counts = df['error'].value_counts().sort_index()
    error_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue', alpha=0.7)
    axes[0, 0].set_xlabel('Error Magnitude')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Error Magnitudes')

    # Errors by capacity
    cap_errors = []
    for cap in sorted(df['node1_capacity'].unique()):
        cap_data = df[(df['node1_capacity'] == cap) | (df['node2_capacity'] == cap)]
        cap_errors.append({
            'capacity': cap,
            'error_rate': (cap_data['error'] > 0).mean(),
            'count': len(cap_data)
        })

    cap_df = pd.DataFrame(cap_errors)
    axes[0, 1].bar(cap_df['capacity'], cap_df['error_rate'], color='lightcoral', alpha=0.7)
    axes[0, 1].set_xlabel('Node Capacity')
    axes[0, 1].set_ylabel('Error Rate')
    axes[0, 1].set_title('Error Rate by Node Capacity')
    axes[0, 1].set_xticks(cap_df['capacity'])

    # Prediction confidence
    correct = df[df['error'] == 0]
    incorrect = df[df['error'] > 0]

    axes[1, 0].hist(correct['pred_prob'], alpha=0.7, label='Correct', bins=20, density=True)
    axes[1, 0].hist(incorrect['pred_prob'], alpha=0.7, label='Incorrect', bins=20, density=True)
    axes[1, 0].set_xlabel('Prediction Confidence')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Confidence Distribution')
    axes[1, 0].legend()

    # Perfect vs imperfect puzzles
    perfect_stats = df.groupby('puzzle_idx').agg({
        'is_perfect': 'first',
        'puzzle_size': 'first',
        'error': 'sum'
    })

    perfect_stats.groupby('is_perfect')['puzzle_size'].plot(kind='hist', alpha=0.7, ax=axes[1, 1],
                                                             label=['Imperfect', 'Perfect'] if axes[1, 1].get_legend() is None else ['', ''])
    axes[1, 1].set_xlabel('Puzzle Size')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Puzzle Size Distribution')
    axes[1, 1].legend()

    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/error_patterns.png", dpi=150, bbox_inches='tight')
        plt.close()

    return fig, axes

if __name__ == "__main__":
    main()
