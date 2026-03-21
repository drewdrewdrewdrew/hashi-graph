"""Callback-driven REINFORCE trainer aligned with ``BaseTrainer`` / engine callbacks."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from tqdm import tqdm

from hashi_puzzle_solver.rl.config import RLConfig
from hashi_puzzle_solver.rl.loader import build_rl_model, load_rl_puzzles
from hashi_puzzle_solver.rl.reinforce import evaluate, train_one_update
from hashi_puzzle_solver.trainers.base import EpochMetrics


def _rl_config_from_dict(config: dict[str, Any]) -> RLConfig:
    r = config.get("rl") or {}
    names = {f.name for f in fields(RLConfig)}
    return RLConfig(**{k: r[k] for k in names if k in r})


def _resolve_model_dir(config: dict[str, Any], callbacks: list[Any]) -> Path:
    model_dir = Path(config["training"].get("model_dir", "models"))
    for cb in callbacks:
        if hasattr(cb, "model_dir"):
            model_dir = cb.model_dir
            break
    return model_dir


class RLTrainer:
    """REINFORCE loop with the same callback hooks as ``BaseTrainer``."""

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.callbacks = callbacks or []

        self._train_puzzles: list[Any] = load_rl_puzzles(
            config, "train", device
        )
        self._val_puzzles: list[Any] = load_rl_puzzles(config, "val", device)

        if not self._train_puzzles:
            msg = "RLTrainer requires at least one training puzzle"
            raise ValueError(msg)

        edge_dim = int(self._train_puzzles[0].edge_attr.size(1))
        self.model = build_rl_model(config, edge_dim, device)

        rt = config.get("rl_training") or {}
        lr = float(rt.get("policy_learning_rate", 0.0001))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self._rl_config = _rl_config_from_dict(config)
        self._n_updates = int(rt.get("n_updates", 500))
        self._eval_every = int(rt.get("eval_every", 10))
        self._puzzles_per_update = int(rt.get("puzzles_per_update", 8))
        self._max_steps = int(rt.get("max_steps_per_rollout", 200))

        self.best_perfect_accuracy: float = 0.0
        self.current_masking_rate: float = 1.0

    def save_model(self, path: str | Path) -> None:
        """Persist policy weights (same contract as ``BaseTrainer.save_model``)."""
        torch.save(self.model.state_dict(), str(path))

    def train(self, _train_transform: object | None = None) -> None:
        """Run REINFORCE updates; transform arg is ignored (API parity with base)."""
        for callback in self.callbacks:
            callback.on_train_start(self)

        model_dir = _resolve_model_dir(self.config, self.callbacks)
        n_train = len(self._train_puzzles)
        batch_n = min(self._puzzles_per_update, n_train)

        try:
            pbar = tqdm(range(self._n_updates), desc="RL Training", unit="update")
            for update in pbar:
                for callback in self.callbacks:
                    callback.on_epoch_start(self, update)

                idx = torch.randperm(n_train, device="cpu")[:batch_n].tolist()
                batch = [self._train_puzzles[i] for i in idx]

                out = train_one_update(
                    batch,
                    self.model,
                    self.optimizer,
                    self._rl_config,
                    max_steps=self._max_steps,
                )
                train_metrics = EpochMetrics()
                train_metrics.loss = float(out["loss"])

                full_rollout_metrics: dict[str, float] | None = None
                if self._eval_every > 0 and update % self._eval_every == 0:
                    full_rollout_metrics = evaluate(
                        self._val_puzzles,
                        self.model,
                        self._rl_config,
                        max_steps=self._max_steps,
                    )

                # Update progress bar with key metrics
                postfix = {"loss": f"{train_metrics.loss:.3f}"}
                if full_rollout_metrics is not None:
                    postfix["perf_acc"] = f"{full_rollout_metrics['perfect_accuracy']:.3f}"
                    postfix["edge_acc"] = f"{full_rollout_metrics['edge_acc']:.3f}"
                    postfix["avg_ret"] = f"{full_rollout_metrics['avg_return']:.2f}"
                pbar.set_postfix(postfix)

                for callback in self.callbacks:
                    callback.on_epoch_end(
                        self,
                        update,
                        train_metrics,
                        None,
                        full_rollout_metrics,
                    )

                if full_rollout_metrics is not None:
                    pa = full_rollout_metrics["perfect_accuracy"]
                    if pa > self.best_perfect_accuracy:
                        self.best_perfect_accuracy = pa
                        self.save_model(model_dir / "model_best.pt")
                        print(
                            f"New best perfect_accuracy: {pa:.4f}. "
                            f"Model saved to {model_dir / 'model_best.pt'}"
                        )
            pbar.close()
        finally:
            for callback in self.callbacks:
                callback.on_train_end(self)
