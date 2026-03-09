"""Rollout LR tuning script for trained diffusion models."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
import yaml

from hashi_puzzle_solver.data import HashiDataset
from hashi_puzzle_solver.models.config import HashiModelConfig
from hashi_puzzle_solver.models.factory import ModelFactory
from hashi_puzzle_solver.trainers.diffusion import DiffusionTrainer
from hashi_puzzle_solver.utils.common import custom_collate_with_conflicts, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune rollout step LR and step count against a fixed val set.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Directory containing model_best.pt and config.yaml",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="Rollout step LR (overrides config diffusion_step_lr)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Number of rollout steps (overrides config diffusion_max_steps)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap evaluation to N randomly sampled puzzles",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override config batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device override: auto, cpu, cuda, mps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_list_filter(value: object) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [int(value)]


def load_raw_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        msg = f"config.yaml not found in {model_dir}"
        raise FileNotFoundError(msg)
    with config_path.open() as f:
        return yaml.safe_load(f)


def load_model(model_dir: Path, cfg: HashiModelConfig, device: torch.device) -> torch.nn.Module:
    weights_path = model_dir / "model_best.pt"
    if not weights_path.exists():
        msg = f"model_best.pt not found in {model_dir}"
        raise FileNotFoundError(msg)
    model = ModelFactory.create_model(cfg, device)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_dataset(cfg: HashiModelConfig, split: str) -> HashiDataset:
    model_cfg = cfg.model
    data_cfg = cfg.data
    return HashiDataset(
        root=data_cfg.root_dir,
        split=split,
        size=normalize_list_filter(data_cfg.size),
        difficulty=normalize_list_filter(data_cfg.difficulty),
        limit=None,
        use_degree=model_cfg.use_degree,
        use_meta_node=model_cfg.use_global_meta_node,
        use_row_col_meta=model_cfg.use_row_col_meta,
        use_meta_mesh=model_cfg.use_meta_mesh,
        use_meta_row_col_edges=model_cfg.use_meta_row_col_edges,
        use_distance=model_cfg.use_distance,
        use_edge_labels_as_features=model_cfg.use_edge_labels_as_features,
        use_closeness_centrality=model_cfg.use_closeness_centrality,
        use_conflict_edges=model_cfg.use_conflict_edges,
        use_capacity=model_cfg.use_capacity,
        use_structural_degree=model_cfg.use_structural_degree,
        use_structural_degree_nsew=model_cfg.use_structural_degree_nsew,
        use_unused_capacity=model_cfg.use_unused_capacity,
        use_conflict_status=model_cfg.use_conflict_status,
        use_articulation_points=model_cfg.use_articulation_points,
        use_cut_edges=model_cfg.use_cut_edges,
        use_spectral_features=model_cfg.use_spectral_features,
        use_potential_crossing=model_cfg.use_potential_crossing,
        use_component_meta=model_cfg.use_component_meta,
        use_continuous_edge_labels=model_cfg.use_continuous_edge_labels,
        use_categorical_edge_types=model_cfg.use_categorical_edge_types,
    )


def build_checkpoints(steps: int) -> list[int]:
    """Generate checkpoints: step 1, every multiple of 5 up to steps, and steps."""
    multiples_of_5 = range(5, steps, 5)
    pts = {1, *multiples_of_5, steps}
    return sorted(pts)


def print_results(
    results: dict,
    args: argparse.Namespace,
    flush_first_step: bool,
    n_puzzles: int,
) -> None:
    checkpoints = build_checkpoints(args.steps)
    print()
    print("=" * 62)
    print(f"ROLLOUT TUNE  lr={args.lr}  steps={args.steps}  split={args.split}  N={n_puzzles}")
    print(f"use_adaptive_sampler=False  flush_first_step={flush_first_step}")
    print("=" * 62)
    print(f"{'step':>6}  {'perfect_acc':>12}")
    print("-" * 22)
    for k in checkpoints:
        val = results.get(f"perfect_acc_k{k}", 0.0)
        print(f"{k:>6}  {val:>12.4f}")
    print("-" * 22)
    print(f"{'edge_acc':>6}  {results.get('accuracy', 0.0):>12.4f}")
    print()


def main() -> None:
    args = parse_args()

    if not args.model_dir.exists():
        msg = f"Model directory not found: {args.model_dir}"
        raise FileNotFoundError(msg)

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    raw_cfg = load_raw_config(args.model_dir)

    # Apply overrides to training config
    raw_cfg["training"]["diffusion_step_lr"] = args.lr
    raw_cfg["training"]["diffusion_max_steps"] = args.steps
    raw_cfg["training"]["use_adaptive_sampler"] = False
    if args.batch_size is not None:
        raw_cfg["training"]["batch_size"] = args.batch_size

    flush_first_step = raw_cfg["training"].get("flush_first_step", False)

    # HashiModelConfig dataclass is needed for dataset + model construction
    cfg = HashiModelConfig.from_dict(raw_cfg)

    print(f"Loading model from: {args.model_dir / 'model_best.pt'}")
    model = load_model(args.model_dir, cfg, device)

    print(f"Loading {args.split} split...")
    dataset = build_dataset(cfg, args.split)

    indices = list(range(len(dataset)))
    if args.limit is not None and args.limit < len(dataset):
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(indices, args.limit))
    n_puzzles = len(indices)
    print(f"Evaluating {n_puzzles} puzzles")

    subset = torch.utils.data.Subset(dataset, indices)
    batch_size = int(raw_cfg["training"]["batch_size"])
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_with_conflicts,
    )

    # Instantiate trainer without calling _setup() — inject the loaded model directly
    trainer = DiffusionTrainer(config=raw_cfg, device=device)
    trainer.model = model

    checkpoints = build_checkpoints(args.steps)
    print(f"Running rollout: lr={args.lr}, steps={args.steps}, checkpoints={checkpoints}")
    results = trainer.run_rollout(loader, max_steps=args.steps, checkpoints=checkpoints)

    print_results(results, args, flush_first_step, n_puzzles)


if __name__ == "__main__":
    main()
