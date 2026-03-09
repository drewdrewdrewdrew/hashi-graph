#!/usr/bin/env python3
"""Sweep component_merge_margin over a fixed grid, all other params held constant.

Runs each margin value sequentially using the hierarchical config as the base,
with a reduced dataset (500 train samples) and short training schedule (50 epochs).

Usage:
    cd /path/to/hashi-graph
    PYTHONPATH=$PYTHONPATH:$(pwd)/src2 uv run python scripts/sweep_merge_margin.py
"""

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

BASE_CONFIG = Path("configs/diffusion_solver_hierarchical.yaml")
MARGINS = [0.2, 0.3, 0.5, 0.7, 0.9]


def main() -> None:
    with BASE_CONFIG.open() as f:
        base = yaml.safe_load(f)

    for margin in MARGINS:
        cfg = deepcopy(base)

        cfg["model"]["component_merge_margin"] = margin
        cfg["data"]["train_limit"] = 500
        cfg["training"]["epochs"] = 50
        cfg["training"]["eval_interval"] = 5

        cfg_path = Path(f"configs/sweep_margin_{margin}.yaml")
        with cfg_path.open("w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        print(f"\n{'='*60}")
        print(f"  margin={margin}  →  {cfg_path}")
        print(f"{'='*60}\n")

        result = subprocess.run(
            [sys.executable, "-m", "hashi_puzzle_solver.train", "--config", str(cfg_path)],
            check=False,
        )
        if result.returncode != 0:
            print(f"[WARN] margin={margin} exited with code {result.returncode}")

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
