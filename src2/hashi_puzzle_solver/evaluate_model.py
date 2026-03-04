"""Diagnostic failure analysis script for trained diffusion models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import yaml

from .data import HashiDataset
from .models.config import HashiModelConfig
from .models.factory import ModelFactory
from .models.features import EdgeFeatureManager
from .utils.common import custom_collate_with_conflicts, get_device
from .utils.diffusion_utils import inject_continuous_noise
from .utils.train_utils import get_edge_batch_indices


PRESET_ORDER = [
    "pure_noise",
    "low_signal",
    "mid_signal",
    "high_signal",
    "near_clean",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostic single-pass inference across fixed noise presets "
            "and report per-puzzle pass/fail structure."
        ),
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Directory containing model_latest.pt and config.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: val)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible preset sweeps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device override (auto, cpu, cuda, mps)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set RNG seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_list_filter(value: Any) -> list[int] | None:
    """Convert config scalar/list to a list or None."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [int(value)]


def load_model_config(model_dir: Path) -> HashiModelConfig:
    """Load config.yaml from model directory into dataclass config."""
    config_path = model_dir / "config.yaml"
    with config_path.open() as f:
        raw_cfg = yaml.safe_load(f)
    return HashiModelConfig.from_dict(raw_cfg)


def load_model(model_dir: Path, cfg: HashiModelConfig, device: torch.device) -> torch.nn.Module:
    """Load model weights from model_latest.pt."""
    model = ModelFactory.create_model(cfg, device)
    checkpoint = torch.load(model_dir / "model_latest.pt", map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_dataset(cfg: HashiModelConfig, split: str) -> HashiDataset:
    """Build full split dataset (limit=None) with config-aligned feature flags."""
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


def build_noise_presets(training_cfg: dict[str, Any]) -> list[dict[str, float | str]]:
    """Build fixed sweep presets from training noise scales."""
    sigma_max = float(training_cfg.get("sigma_max", 2.0))
    scale_min = float(training_cfg.get("scale_min", 1.0))
    scale_max = float(training_cfg.get("scale_max", 1.0))
    scale_mid = (scale_min + scale_max) / 2.0
    return [
        {"name": "pure_noise", "alpha": 0.0, "sigma": sigma_max, "scale": scale_mid},
        {"name": "low_signal", "alpha": 0.25, "sigma": sigma_max * 0.75, "scale": scale_mid},
        {"name": "mid_signal", "alpha": 0.5, "sigma": sigma_max * 0.5, "scale": scale_mid},
        {"name": "high_signal", "alpha": 0.75, "sigma": sigma_max * 0.25, "scale": scale_mid},
        {"name": "near_clean", "alpha": 0.95, "sigma": 0.1, "scale": scale_mid},
    ]


def extract_puzzle_properties(data: Data, puzzle_file: str) -> dict[str, Any]:
    """Extract structural puzzle properties from Data."""
    node_type = data.node_type
    is_puzzle_node = (node_type >= 1) & (node_type <= 8)
    puzzle_node_types = node_type[is_puzzle_node]

    if is_puzzle_node.any():
        valid_pos = data.pos[is_puzzle_node]
        max_x = int(valid_pos[:, 0].max().item())
        max_y = int(valid_pos[:, 1].max().item())
        grid_size = max(max_x, max_y) + 1
    else:
        grid_size = 0

    edge_count = int(data.edge_mask.sum().item() // 2)
    node_count = int(is_puzzle_node.sum().item())
    max_capacity = int(puzzle_node_types.max().item()) if node_count > 0 else 0
    num_conflicts = len(getattr(data, "edge_conflicts", []) or [])

    row: dict[str, Any] = {
        "puzzle_file": puzzle_file,
        "grid_size": grid_size,
        "node_count": node_count,
        "edge_count": edge_count,
        "edge_node_ratio": (edge_count / node_count) if node_count > 0 else 0.0,
        "max_capacity": max_capacity,
        "num_conflicts": num_conflicts,
    }
    for cap in range(1, 9):
        row[f"cap_{cap}"] = int((puzzle_node_types == cap).sum().item())
    return row


def compute_graph_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
) -> list[dict[str, float | int | bool]]:
    """Compute per-graph perfect/wrong/accuracy metrics."""
    metrics: list[dict[str, float | int | bool]] = []
    for graph_idx in range(num_graphs):
        mask = edge_batch == graph_idx
        graph_targets = targets[mask]
        graph_preds = predictions[mask]
        wrong = int((graph_preds != graph_targets).sum().item())
        total = int(mask.sum().item())
        metrics.append(
            {
                "is_perfect": wrong == 0,
                "num_wrong_edges": wrong,
                "edge_accuracy": (1.0 - wrong / total) if total > 0 else 0.0,
            },
        )
    return metrics


@torch.no_grad()
def run_preset(
    model: torch.nn.Module,
    loader: DataLoader,
    preset: dict[str, float | str],
    bridge_logits_idx: int,
    model_config_dict: dict[str, Any],
    device: torch.device,
) -> list[dict[str, float | int | bool]]:
    """Run one fixed-noise preset across full loader."""
    all_results: list[dict[str, float | int | bool]] = []
    alpha = float(preset["alpha"])
    sigma = float(preset["sigma"])
    scale = float(preset["scale"])
    for batch in loader:
        batch = batch.to(device)
        noisy_batch = inject_continuous_noise(
            batch,
            alpha=alpha,
            sigma=sigma,
            scale=scale,
            bridge_logits_idx=bridge_logits_idx,
            model_config=model_config_dict,
            device=device,
        )
        logits = model(
            noisy_batch.x,
            noisy_batch.edge_index,
            edge_attr=noisy_batch.edge_attr,
            edge_type=getattr(noisy_batch, "edge_type", None),
            batch=noisy_batch.batch,
            node_type=noisy_batch.node_type,
            input_noise=torch.tensor([[sigma, alpha]], device=device).repeat(
                noisy_batch.num_graphs, 1
            ),
        )
        edge_mask = noisy_batch.edge_mask
        masked_logits = logits[edge_mask]
        targets = noisy_batch.y[edge_mask]
        edge_batch = get_edge_batch_indices(noisy_batch)[edge_mask]
        predictions = masked_logits.argmax(dim=-1)
        all_results.extend(
            compute_graph_metrics(
                predictions=predictions,
                targets=targets,
                edge_batch=edge_batch,
                num_graphs=noisy_batch.num_graphs,
            ),
        )
    return all_results


def print_report(df: pd.DataFrame) -> None:
    """Print summary and failure-structure analysis tables."""
    print("\n" + "=" * 80)
    print("NOISE PRESET SUMMARY")
    print("=" * 80)
    summary_rows = []
    for preset in PRESET_ORDER:
        summary_rows.append(
            {
                "preset": preset,
                "perfect_acc": float(df[f"{preset}_perfect"].mean()),
                "edge_acc": float(df[f"{preset}_edge_acc"].mean()),
            },
        )
    print(pd.DataFrame(summary_rows).to_string(index=False))

    structural_cols = [
        "grid_size",
        "node_count",
        "edge_count",
        "edge_node_ratio",
        "max_capacity",
        "num_conflicts",
    ] + [f"cap_{cap}" for cap in range(1, 9)]

    pure_fail_mask = ~df["pure_noise_perfect"]
    failed = df[pure_fail_mask]
    passed = df[~pure_fail_mask]

    print("\n" + "=" * 80)
    print("FAILED PUZZLE PROPERTIES (PURE NOISE)")
    print("=" * 80)
    if failed.empty:
        print("No pure_noise failures.")
    else:
        print(failed[structural_cols].describe().to_string())

    print("\n" + "=" * 80)
    print("PASSED VS FAILED (PURE NOISE): MEAN STRUCTURE")
    print("=" * 80)
    if failed.empty or passed.empty:
        print("Comparison unavailable (need both pass and fail groups).")
    else:
        comparison = pd.DataFrame(
            {"failed_mean": failed[structural_cols].mean(), "passed_mean": passed[structural_cols].mean()},
        )
        print(comparison.to_string())

    print("\n" + "=" * 80)
    print("CAPACITY PRESENCE IN PURE NOISE FAILURES")
    print("=" * 80)
    cap_rows = []
    for cap in range(1, 9):
        col = f"cap_{cap}"
        overall_rate = float((df[col] > 0).mean() * 100.0)
        fail_rate = float((failed[col] > 0).mean() * 100.0) if not failed.empty else 0.0
        cap_rows.append(
            {
                "capacity": cap,
                "failure_presence_pct": round(fail_rate, 2),
                "overall_presence_pct": round(overall_rate, 2),
            },
        )
    print(pd.DataFrame(cap_rows).to_string(index=False))


def run_diagnostic(args: argparse.Namespace) -> pd.DataFrame:
    """Run diagnostic sweep and return one-row-per-puzzle DataFrame."""
    model_dir = args.model_dir
    if not model_dir.exists():
        msg = f"Model directory not found: {model_dir}"
        raise FileNotFoundError(msg)

    cfg = load_model_config(model_dir)
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading model from: {model_dir}")
    model = load_model(model_dir, cfg, device)

    print(f"Loading full {args.split} split...")
    dataset = build_dataset(cfg, args.split)
    print(f"Loaded {len(dataset)} puzzles")

    edge_fm = EdgeFeatureManager(cfg.model)
    if not edge_fm.has_feature("bridge_logits"):
        msg = "Model config must enable use_continuous_edge_labels for diagnostic sweep."
        raise ValueError(msg)
    bridge_logits_idx = edge_fm.get_idx("bridge_logits")

    rows = [
        extract_puzzle_properties(dataset[i], dataset._raw_filenames[i]) for i in range(len(dataset))
    ]
    df = pd.DataFrame(rows)

    batch_size = int(cfg.training.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_with_conflicts,
    )
    model_config_dict = asdict(cfg.model)
    presets = build_noise_presets(asdict(cfg.training))
    for preset_idx, preset in enumerate(presets):
        preset_name = str(preset["name"])
        set_seed(args.seed + preset_idx)
        print(
            f"Running preset={preset_name} alpha={preset['alpha']:.2f} "
            f"sigma={preset['sigma']:.2f} scale={preset['scale']:.2f}",
        )
        preset_results = run_preset(
            model=model,
            loader=loader,
            preset=preset,
            bridge_logits_idx=bridge_logits_idx,
            model_config_dict=model_config_dict,
            device=device,
        )
        if len(preset_results) != len(df):
            msg = (
                f"Result count mismatch for preset {preset_name}: "
                f"{len(preset_results)} vs dataset size {len(df)}"
            )
            raise RuntimeError(msg)
        df[f"{preset_name}_perfect"] = [bool(r["is_perfect"]) for r in preset_results]
        df[f"{preset_name}_wrong_edges"] = [int(r["num_wrong_edges"]) for r in preset_results]
        df[f"{preset_name}_edge_acc"] = [float(r["edge_accuracy"]) for r in preset_results]
    return df


def main() -> None:
    """Entry point."""
    args = parse_args()
    df = run_diagnostic(args)
    print_report(df)


if __name__ == "__main__":
    main()
