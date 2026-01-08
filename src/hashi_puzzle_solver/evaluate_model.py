"""
Auxiliary script to evaluate a trained model on any dataset.

Calculates edge-level accuracy and perfect puzzle accuracy.

Usage:
    python -m src.evaluate_model --model_path <path> --config <path>
    [--split <train|val|test>]
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import yaml

from .data import HashiDataset
from .models import (
    GATEdgeClassifier,
    GCNEdgeClassifier,
    GINEEdgeClassifier,
    TransformerEdgeClassifier,
)
from .train_utils import (
    calculate_batch_perfect_puzzles,
    evaluate_puzzle,
    get_edge_batch_indices,
)


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def get_device(device_config: str) -> torch.device:
    """Determine the compute device based on config and availability."""
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_config)


def load_model(
    model_path: str,
    model_config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained model from a checkpoint."""
    model_type = model_config.get("type", "gcn").lower()
    use_degree = model_config.get("use_degree", False)
    use_meta_node = model_config.get("use_meta_node", False)
    use_row_col_meta = model_config.get("use_row_col_meta", False)
    use_closeness_centrality = model_config.get("use_closeness_centrality", False)
    use_conflict_edges = model_config.get("use_conflict_edges", False)
    use_edge_labels_as_features = model_config.get("use_edge_labels_as_features", False)

    # Determine edge dimension
    edge_dim = 3  # base: [inv_dx, inv_dy, is_meta]
    if use_conflict_edges:
        edge_dim += 1
    if use_edge_labels_as_features:
        edge_dim += 2

    # Initialize model
    if model_type == "gcn":
        model = GCNEdgeClassifier(
            node_embedding_dim=model_config["node_embedding_dim"],
            hidden_channels=model_config["hidden_channels"],
            num_layers=model_config["num_layers"],
            dropout=model_config.get("dropout", 0.25),
            use_degree=use_degree,
            use_meta_node=use_meta_node,
            use_closeness=use_closeness_centrality,
        ).to(device)
    elif model_type == "gat":
        model = GATEdgeClassifier(
            node_embedding_dim=model_config["node_embedding_dim"],
            hidden_channels=model_config["hidden_channels"],
            num_layers=model_config["num_layers"],
            heads=model_config.get("heads", 8),
            dropout=model_config.get("dropout", 0.25),
            use_degree=use_degree,
            use_meta_node=use_meta_node,
            use_row_col_meta=use_row_col_meta,
            edge_dim=edge_dim,
            use_closeness=use_closeness_centrality,
        ).to(device)
    elif model_type == "gine":
        model = GINEEdgeClassifier(
            node_embedding_dim=model_config["node_embedding_dim"],
            hidden_channels=model_config["hidden_channels"],
            num_layers=model_config["num_layers"],
            dropout=model_config.get("dropout", 0.25),
            use_degree=use_degree,
            use_meta_node=use_meta_node,
            use_row_col_meta=use_row_col_meta,
            edge_dim=edge_dim,
            use_closeness=use_closeness_centrality,
        ).to(device)
    elif model_type == "transformer":
        model = TransformerEdgeClassifier(
            node_embedding_dim=model_config["node_embedding_dim"],
            hidden_channels=model_config["hidden_channels"],
            num_layers=model_config["num_layers"],
            heads=model_config.get("heads", 4),
            dropout=model_config.get("dropout", 0.25),
            use_degree=use_degree,
            use_meta_node=use_meta_node,
            use_row_col_meta=use_row_col_meta,
            edge_dim=edge_dim,
            use_closeness=use_closeness_centrality,
        ).to(device)
    else:
        error_msg = f"Unknown model type: {model_type}"
        raise ValueError(error_msg)

    feature_flags = {
        "use_degree": use_degree,
        "use_meta_node": use_meta_node,
        "use_conflict_edges": use_conflict_edges,
        "use_closeness_centrality": use_closeness_centrality,
    }

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as exc:
        error_msg = (
            f"Failed to load checkpoint '{model_path}' with current config "
            f"(type={model_type}, features={feature_flags}). "
            f"Please verify that the model structure matches your config. "
            f"Original error: {exc}"
        )
        raise RuntimeError(error_msg) from exc

    model.eval()

    return model


@torch.no_grad()
def evaluate_model_batched(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a model on a dataset using batched processing (faster).

    Returns
    -------
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_edges = 0
    perfect_puzzle_stats = []
    criterion = torch.nn.CrossEntropyLoss()

    # Track per-class accuracy
    class_correct = [0, 0, 0]  # For classes 0, 1, 2
    class_total = [0, 0, 0]

    iterator = tqdm(loader, desc="Evaluating") if verbose else loader
    for data in iterator:
        data = data.to(device)

        # Pass edge_attr if present
        edge_attr = getattr(data, "edge_attr", None)
        logits = model(data.x, data.edge_index, edge_attr=edge_attr)

        # Use edge_mask to select only original puzzle edges
        logits_original = logits[data.edge_mask]
        loss = criterion(logits_original, data.y)

        total_loss += loss.item() * data.num_graphs
        pred = logits_original.argmax(dim=-1)
        correct_predictions += (pred == data.y).sum().item()
        total_edges += data.edge_mask.sum().item()

        # Track per-class accuracy
        for class_idx in range(3):
            class_mask = data.y == class_idx
            class_correct[class_idx] += (
                (pred[class_mask] == data.y[class_mask]).sum().item()
            )
            class_total[class_idx] += class_mask.sum().item()

        # Calculate perfect puzzle accuracy
        edge_batch = get_edge_batch_indices(data)
        edge_batch_original = edge_batch[data.edge_mask]
        _, num_perfect, num_total = calculate_batch_perfect_puzzles(
            logits_original,
            data.y,
            torch.ones_like(data.edge_mask[data.edge_mask], dtype=torch.bool),
            edge_batch_original,
        )
        perfect_puzzle_stats.append((num_perfect, num_total))

    # Calculate overall metrics
    avg_loss = total_loss / len(loader.dataset)
    edge_accuracy = correct_predictions / total_edges if total_edges > 0 else 0.0

    total_perfect = sum(perfect for perfect, _ in perfect_puzzle_stats)
    total_puzzles = sum(total for _, total in perfect_puzzle_stats)
    perfect_puzzle_accuracy = (
        total_perfect / total_puzzles if total_puzzles > 0 else 0.0
    )

    # Calculate per-class accuracy
    per_class_acc = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(3)
    ]

    return {
        "loss": avg_loss,
        "edge_accuracy": edge_accuracy,
        "perfect_puzzle_accuracy": perfect_puzzle_accuracy,
        "total_puzzles": total_puzzles,
        "perfect_puzzles": total_perfect,
        "total_edges": total_edges,
        "correct_edges": correct_predictions,
        "per_class_accuracy": per_class_acc,
        "class_total": class_total,
    }


@torch.no_grad()
def evaluate_model_puzzle_by_puzzle(
    model: torch.nn.Module,
    dataset: HashiDataset,
    device: torch.device,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a model puzzle-by-puzzle (cleaner, works directly with dataset).

    Args:
        model: Trained model
        dataset: HashiDataset instance
        device: torch device
        verbose: Show progress bar

    Returns
    -------
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_edges = 0
    correct_edges = 0
    perfect_puzzles = 0

    # Track per-class accuracy
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    iterator = (
        tqdm(range(len(dataset)), desc="Evaluating") if verbose else range(len(dataset))
    )

    for idx in iterator:
        data = dataset[idx]
        result = evaluate_puzzle(model, data, device)

        # Accumulate statistics
        predictions = result["predictions"].to(device)
        targets = result["targets"].to(device)

        # Calculate loss
        data = data.to(device)
        edge_attr = getattr(data, "edge_attr", None)
        logits = model(data.x, data.edge_index, edge_attr=edge_attr)
        original_mask = (
            data.edge_mask
            if hasattr(data, "edge_mask")
            else torch.ones(logits.size(0), dtype=torch.bool)
        )
        logits_original = logits[original_mask]
        loss = criterion(logits_original, targets)
        total_loss += loss.item()

        # Edge accuracy
        correct_edges += (predictions == targets).sum().item()
        total_edges += len(targets)

        # Perfect puzzle
        if result["is_perfect"]:
            perfect_puzzles += 1

        # Per-class accuracy
        for class_idx in range(3):
            class_mask = targets == class_idx
            class_correct[class_idx] += (
                (predictions[class_mask] == targets[class_mask]).sum().item()
            )
            class_total[class_idx] += class_mask.sum().item()

    # Calculate overall metrics
    num_puzzles = len(dataset)
    avg_loss = total_loss / num_puzzles if num_puzzles > 0 else 0.0
    edge_accuracy = correct_edges / total_edges if total_edges > 0 else 0.0
    perfect_puzzle_accuracy = perfect_puzzles / num_puzzles if num_puzzles > 0 else 0.0

    # Calculate per-class accuracy
    per_class_acc = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(3)
    ]

    return {
        "loss": avg_loss,
        "edge_accuracy": edge_accuracy,
        "perfect_puzzle_accuracy": perfect_puzzle_accuracy,
        "total_puzzles": num_puzzles,
        "perfect_puzzles": perfect_puzzles,
        "total_edges": total_edges,
        "correct_edges": correct_edges,
        "per_class_accuracy": per_class_acc,
        "class_total": class_total,
    }


def main() -> None:
    """Evaluate a trained model on a dataset."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on any dataset.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file used for training",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: val)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: use config value)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="puzzle",
        choices=["puzzle", "batch"],
        help="Evaluation mode: 'puzzle' for puzzle-by-puzzle (cleaner), "
             "'batch' for batched (faster)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["training"]

    # Set device
    device = get_device(
        args.device if args.device != "auto" else train_config.get("device", "auto"),
    )
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, model_config, device)
    print(f"Loaded {model_config.get('type', 'gcn').upper()} model")

    # Load dataset
    root_path = Path(data_config["root_dir"])
    size_filter = data_config.get("size") or None
    difficulty_filter = data_config.get("difficulty") or None
    limit = data_config.get("limit") or None

    # Get feature flags from model config
    use_degree = model_config.get("use_degree", False)
    use_meta_node = model_config.get("use_meta_node", False)
    use_row_col_meta = model_config.get("use_row_col_meta", False)
    use_distance = model_config.get("use_distance", False)
    use_edge_labels_as_features = model_config.get("use_edge_labels_as_features", False)
    use_closeness_centrality = model_config.get("use_closeness_centrality", False)
    use_conflict_edges = model_config.get("use_conflict_edges", False)

    print(f"Loading {args.split} dataset...")
    dataset = HashiDataset(
        root=root_path,
        split=args.split,
        size=size_filter,
        difficulty=difficulty_filter,
        limit=limit,
        use_degree=use_degree,
        use_meta_node=use_meta_node,
        use_row_col_meta=use_row_col_meta,
        use_distance=use_distance,
        use_edge_labels_as_features=use_edge_labels_as_features,
        use_closeness_centrality=use_closeness_centrality,
        use_conflict_edges=use_conflict_edges,
    )
    print(f"Loaded {len(dataset)} puzzles")

    # Evaluate
    print(f"\nEvaluating model using {args.mode} mode...")
    if args.mode == "puzzle":
        # Puzzle-by-puzzle evaluation (cleaner, works directly with dataset)
        results = evaluate_model_puzzle_by_puzzle(model, dataset, device)
    else:
        # Batched evaluation (faster for large datasets)
        batch_size = args.batch_size or train_config["batch_size"]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        results = evaluate_model_batched(model, loader, device)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset Split:           {args.split}")
    print(f"Total Puzzles:           {results['total_puzzles']}")
    print(f"Total Edges:             {results['total_edges']}")
    print("-" * 60)
    print(f"Loss:                    {results['loss']:.4f}")
    print(
        f"Edge Accuracy:           {results['edge_accuracy']:.4f} "
        f"({results['correct_edges']}/{results['total_edges']})",
    )
    print(
        f"Perfect Puzzle Accuracy: {results['perfect_puzzle_accuracy']:.4f} "
        f"({results['perfect_puzzles']}/{results['total_puzzles']})",
    )
    print("-" * 60)
    print("Per-Class Accuracy:")
    for i in range(3):
        print(
            f"  Class {i}: {results['per_class_accuracy'][i]:.4f} "
            f"({results['class_total'][i]} edges)",
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
