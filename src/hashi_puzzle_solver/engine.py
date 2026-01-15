"""
Core training engine for Hashi GNN.

Centralizes model creation, dataset loading, and training loop components.
"""

from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ar_engine import ARTrainer
from .data import HashiDataset, HashiDatasetCache
from .diffusion_engine import DiffusionTrainer
from .losses import compute_combined_loss
from .masking import MaskingStrategy
from .models.factory import ModelFactory
from .train_utils import calculate_batch_perfect_puzzles, get_edge_batch_indices
from .utils import custom_collate_with_conflicts


class EpochMetrics:
    """Container for metrics returned from run_epoch."""

    def __init__(self) -> None:
        self.loss: float = 0.0
        self.accuracy: float = 0.0
        self.perfect_accuracy: float = 0.0
        self.ce_loss: float = 0.0
        self.degree_loss: float = 0.0
        self.crossing_loss: float = 0.0
        self.verify_loss: float = 0.0
        self.noise_loss: float = 0.0
        self.sigma_loss: float = 0.0
        self.alpha_loss: float = 0.0
        self.verify_balanced_acc: float = 0.0
        self.verify_recall_pos: float = 0.0
        self.verify_recall_neg: float = 0.0

    def to_tuple(
        self,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:
        """Return metrics as tuple for backward compatibility."""
        return (
            self.loss,
            self.ce_loss,
            self.degree_loss,
            self.crossing_loss,
            self.verify_loss,
            self.sigma_loss,
            self.alpha_loss,
            self.accuracy,
            self.perfect_accuracy,
            self.verify_balanced_acc,
            self.verify_recall_pos,
            self.verify_recall_neg,
        )


class EarlyStopper:
    """Utility to signal when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 1,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def early_stop(self, current_value: float) -> bool:
        """Return True once the monitored metric fails to improve."""
        if self.mode == "min":
            if current_value < (self.best_value - self.min_delta):
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == "max"
            if current_value > (self.best_value + self.min_delta):
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1

        return self.counter >= self.patience


class Trainer:
    """
    Encapsulate training logic.

    Provides a unified interface for model initialization, dataloader creation,
    and epoch execution.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        callbacks: list[Any] | None = None,
    ):
        if callbacks is None:
            callbacks = []
        self.config = config
        self.device = device
        self.callbacks = callbacks
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.masking_strategy = MaskingStrategy(config)
        self.current_masking_rate = 0.0
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.best_monitored_value = None

    def _setup(self, train_transform: object | None = None) -> None:
        """Set up model, optimizer, and data loaders."""
        self.model = ModelFactory.create_model(self.config, self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
        )
        self.train_loader = self.create_dataloader(
            split="train",
            transform=train_transform,
        )
        self.val_loader = self.create_dataloader(split="val")

    def train(self, train_transform: object | None = None) -> None:
        """Run main training loop."""
        self._setup(train_transform)

        epochs = self.config["training"]["epochs"]
        mode = self.config["training"].get("mode", "one-shot").lower()
        eval_interval = self.config["training"].get("eval_interval", 1)
        accumulation_steps = self.config["training"].get("accumulation_steps", 1)
        early_stopping_config = self.config["training"].get("early_stopping", {})

        monitor = early_stopping_config.get("monitor", "loss")
        # Strip 'val_' prefix if present (we always monitor validation anyway)
        metric_key = monitor.replace("val_", "")
        if metric_key == "perfect_acc":
            metric_key = "perfect_accuracy"

        # Determine stop_mode (max for accuracies, min for losses)
        stop_mode = "max" if "acc" in metric_key.lower() else "min"
        self.best_monitored_value = (
            float("inf") if stop_mode == "min" else float("-inf")
        )

        early_stopper = EarlyStopper(
            monitor=monitor,
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 0.0),
            mode=stop_mode,
        )

        # Setup AR or Diffusion if needed
        ar_trainer = None
        diffusion_trainer = None
        if mode == "ar":
            ar_trainer = ARTrainer(self.model, self.config, self.device)
        elif mode in ["diff-discrete", "diff-cont"]:
            diffusion_trainer = DiffusionTrainer(self.model, self.config, self.device)

        for callback in self.callbacks:
            callback.on_train_start(self)

        try:
            for epoch in range(1, epochs + 1):
                for callback in self.callbacks:
                    callback.on_epoch_start(self, epoch)

                if mode == "ar":
                    self.current_masking_rate = self.masking_strategy.get_rate(
                        epoch,
                        epochs,
                    )
                    ar_results = ar_trainer.run_epoch(
                        self.train_loader,
                        epoch=epoch,
                        total_epochs=epochs,
                        optimizer=self.optimizer,
                        training=True,
                    )
                    train_metrics = EpochMetrics()
                    train_metrics.loss = ar_results["loss"]
                    train_metrics.ce_loss = ar_results["ce_loss"]
                    train_metrics.degree_loss = ar_results["degree_loss"]
                    train_metrics.crossing_loss = ar_results["crossing_loss"]
                    train_metrics.accuracy = ar_results["accuracy"]
                    train_metrics.perfect_accuracy = ar_results[
                        "perfect_accuracy"
                    ]
                    train_metrics.verify_loss = ar_results["verify_loss"]
                    train_metrics.verify_balanced_acc = ar_results[
                        "verify_balanced_acc"
                    ]
                    train_metrics.verify_recall_pos = ar_results[
                        "verify_recall_pos"
                    ]
                    train_metrics.verify_recall_neg = ar_results[
                        "verify_recall_neg"
                    ]
                elif mode in ["diff-discrete", "diff-cont"]:
                    self.current_masking_rate = self.masking_strategy.get_rate(
                        epoch,
                        epochs,
                    )
                    diff_results = diffusion_trainer.run_epoch(
                        self.train_loader,
                        epoch=epoch,
                        total_epochs=epochs,
                        optimizer=self.optimizer,
                        training=True,
                        noise_rate=self.current_masking_rate,
                    )
                    train_metrics = EpochMetrics()
                    train_metrics.loss = diff_results["loss"]
                    train_metrics.ce_loss = diff_results["ce_loss"]
                    train_metrics.degree_loss = diff_results["degree_loss"]
                    train_metrics.crossing_loss = diff_results["crossing_loss"]
                    train_metrics.accuracy = diff_results["accuracy"]
                    train_metrics.perfect_accuracy = diff_results[
                        "perfect_accuracy"
                    ]
                    train_metrics.verify_loss = diff_results["verify_loss"]
                    train_metrics.verify_balanced_acc = diff_results.get(
                        "verify_balanced_acc", 0.0
                    )
                    train_metrics.verify_recall_pos = diff_results.get(
                        "verify_recall_pos", 0.0
                    )
                    train_metrics.verify_recall_neg = diff_results.get(
                        "verify_recall_neg", 0.0
                    )
                    if "sigma_loss" in diff_results:
                        train_metrics.sigma_loss = diff_results["sigma_loss"]
                    if "alpha_loss" in diff_results:
                        train_metrics.alpha_loss = diff_results["alpha_loss"]
                    if "noise_loss" in diff_results:
                        train_metrics.noise_loss = diff_results["noise_loss"]
                else:
                    # Standard One-Shot training
                    self.current_masking_rate = self.masking_strategy.get_rate(
                        epoch,
                        epochs,
                    )

                    train_metrics = self.run_epoch_one_shot(
                        self.model,
                        self.train_loader,
                        training=True,
                        optimizer=self.optimizer,
                        masking_rate=self.current_masking_rate,
                        accumulation_steps=accumulation_steps,
                    )

                # Clear memory after training pass
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()

                val_metrics = None
                full_rollout_metrics = None
                if epoch % eval_interval == 0:
                    if mode == "ar":
                        ar_results_val = ar_trainer.run_epoch(
                            self.val_loader,
                            epoch=epoch,
                            total_epochs=epochs,
                            training=False,
                        )
                        val_metrics = EpochMetrics()
                        val_metrics.loss = ar_results_val["loss"]
                        val_metrics.ce_loss = ar_results_val["ce_loss"]
                        val_metrics.degree_loss = ar_results_val["degree_loss"]
                        val_metrics.crossing_loss = ar_results_val["crossing_loss"]
                        val_metrics.accuracy = ar_results_val["accuracy"]
                        val_metrics.perfect_accuracy = ar_results_val[
                            "perfect_accuracy"
                        ]
                        val_metrics.verify_loss = ar_results_val["verify_loss"]
                        val_metrics.verify_balanced_acc = ar_results_val[
                            "verify_balanced_acc"
                        ]
                        val_metrics.verify_recall_pos = ar_results_val[
                            "verify_recall_pos"
                        ]
                        val_metrics.verify_recall_neg = ar_results_val[
                            "verify_recall_neg"
                        ]
                    elif mode in ["diff-discrete", "diff-cont"]:
                        # Distributionally identical validation for diffusion
                        diff_results_val = diffusion_trainer.run_epoch(
                            self.val_loader,
                            epoch=epoch,
                            total_epochs=epochs,
                            training=False,
                            noise_rate=self.current_masking_rate,
                        )
                        val_metrics = EpochMetrics()
                        val_metrics.loss = diff_results_val["loss"]
                        val_metrics.ce_loss = diff_results_val["ce_loss"]
                        val_metrics.degree_loss = diff_results_val["degree_loss"]
                        val_metrics.crossing_loss = diff_results_val["crossing_loss"]
                        val_metrics.accuracy = diff_results_val["accuracy"]
                        val_metrics.perfect_accuracy = diff_results_val[
                            "perfect_accuracy"
                        ]
                        val_metrics.verify_loss = diff_results_val["verify_loss"]
                        val_metrics.verify_balanced_acc = diff_results_val.get(
                            "verify_balanced_acc", 0.0
                        )
                        val_metrics.verify_recall_pos = diff_results_val.get(
                            "verify_recall_pos", 0.0
                        )
                        val_metrics.verify_recall_neg = diff_results_val.get(
                            "verify_recall_neg", 0.0
                        )
                        # Sigma loss is only for diff-cont
                        if "sigma_loss" in diff_results_val:
                            val_metrics.sigma_loss = diff_results_val["sigma_loss"]
                        if "alpha_loss" in diff_results_val:
                            val_metrics.alpha_loss = diff_results_val["alpha_loss"]
                        if "noise_loss" in diff_results_val:
                            val_metrics.noise_loss = diff_results_val["noise_loss"]

                        # Trigger iterative rollout validation if interval reached
                        training_cfg = self.config["training"]
                        masking_cfg = training_cfg.get("masking", {})

                        eval_rollout_interval = training_cfg.get(
                            "eval_rollout_interval",
                            masking_cfg.get("eval_rollout_interval", 0)
                        )
                        if (
                            eval_rollout_interval > 0
                            and epoch % eval_rollout_interval == 0
                        ):
                            max_steps = training_cfg.get(
                                "diffusion_max_steps",
                                masking_cfg.get("diffusion_max_steps", 20)
                            )
                            full_rollout_metrics = diffusion_trainer.run_rollout(
                                self.val_loader,
                                max_steps=max_steps,
                            )
                    else:
                        # Standard One-Shot validation
                        val_metrics = self.run_epoch_one_shot(
                            self.model,
                            self.val_loader,
                            training=False,
                            masking_rate=1.0,
                        )

                    # Clear memory after validation pass
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                    elif self.device.type == "cuda":
                        torch.cuda.empty_cache()

                for callback in self.callbacks:
                    callback.on_epoch_end(
                        self, epoch, train_metrics, val_metrics, full_rollout_metrics
                    )

                if val_metrics:
                    self.best_val_acc = max(self.best_val_acc, val_metrics.accuracy)
                    self.best_val_loss = min(self.best_val_loss, val_metrics.loss)

                    # Check for improvement in monitored metric
                    current_val = getattr(val_metrics, metric_key)
                    is_better = False
                    if stop_mode == "min":
                        if current_val < (
                            self.best_monitored_value - early_stopper.min_delta
                        ):
                            is_better = True
                    else:  # stop_mode == "max"
                        if current_val > (
                            self.best_monitored_value + early_stopper.min_delta
                        ):
                            is_better = True

                    if is_better:
                        self.best_monitored_value = current_val
                        # Save best model
                        model_dir = Path(
                            self.config["training"].get("model_dir", "models")
                        )
                        # Find the actual run directory if it exists in callbacks
                        for cb in self.callbacks:
                            if hasattr(cb, "model_dir"):
                                model_dir = cb.model_dir
                                break

                        best_path = model_dir / "model_best.pt"
                        torch.save(self.model.state_dict(), str(best_path))
                        print(
                            f"New best {monitor}: {current_val:.4f}. "
                            f"Model weights saved to {best_path}"
                        )

                    # Early Stopping Check
                    if early_stopper.early_stop(current_val):
                        print(
                            f"Early stopping triggered at epoch {epoch} "
                            f"(monitoring {monitor})",
                        )
                        break
        finally:
            for callback in self.callbacks:
                callback.on_train_end(self)

    def create_dataloader(
        self,
        split: str,
        transform: object | None = None,
        use_cache: bool = False,
    ) -> DataLoader:
        """Create a dataloader for the specified split."""
        data_config = self.config["data"]
        model_config = self.config["model"]
        training_config = self.config["training"]

        limit = data_config.get("limit")

        if use_cache:
            dataset = HashiDatasetCache.get_or_create(
                self.config,
                split,
                transform=transform,
            )
        else:
            dataset = HashiDataset(
                root=Path(data_config["root_dir"]),
                split=split,
                size=data_config.get("size"),
                difficulty=data_config.get("difficulty"),
                limit=None,  # REDEFINED: Always index all files for dynamic subsampling
                use_degree=model_config.get("use_degree", False),
                use_meta_node=model_config.get("use_global_meta_node", True),
                use_row_col_meta=model_config.get("use_row_col_meta", False),
                use_meta_mesh=model_config.get("use_meta_mesh", False),
                use_meta_row_col_edges=model_config.get(
                    "use_meta_row_col_edges",
                    False,
                ),
                use_distance=model_config.get("use_distance", False),
                use_edge_labels_as_features=model_config.get(
                    "use_edge_labels_as_features", False,
                ),
                use_closeness_centrality=model_config.get(
                    "use_closeness_centrality", False,
                ),
                use_conflict_edges=model_config.get("use_conflict_edges", False),
                use_capacity=model_config.get("use_capacity", True),
                use_structural_degree=model_config.get("use_structural_degree", True),
                use_structural_degree_nsew=model_config.get(
                    "use_structural_degree_nsew", False,
                ),
                use_unused_capacity=model_config.get("use_unused_capacity", True),
                use_conflict_status=model_config.get("use_conflict_status", True),
                use_articulation_points=model_config.get(
                    "use_articulation_points", False,
                ),
                use_cut_edges=model_config.get("use_cut_edges", False),
                use_spectral_features=model_config.get("use_spectral_features", False),
                use_potential_crossing=model_config.get("use_potential_crossing", False),
                use_component_meta=model_config.get("use_component_meta", False),
                use_continuous_edge_labels=model_config.get(
                    "use_continuous_edge_labels", False
                ),
                transform=transform,
            )

        sampler = None
        shuffle = (split == "train")

        if limit is not None:
            # REDEFINED: limit now means "samples per epoch"
            num_samples = min(int(limit), len(dataset))
            if split == "train":
                from torch.utils.data import RandomSampler
                sampler = RandomSampler(
                    dataset,
                    num_samples=num_samples,
                    replacement=False,
                )
                shuffle = False
            elif split == "val":
                # For validation, use a fixed subset of size 'limit' for consistency
                from torch.utils.data import SubsetRandomSampler
                indices = list(range(num_samples))
                sampler = SubsetRandomSampler(indices)
                shuffle = False

        return DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=training_config.get("num_workers", 0),
            collate_fn=custom_collate_with_conflicts,
            persistent_workers=training_config.get(
                "use_persistent_workers",
                False,
            ),
        )

    def run_epoch_one_shot(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        training: bool = True,
        optimizer: Optimizer | None = None,
        masking_rate: float = 0.0,
        accumulation_steps: int = 1,
    ) -> EpochMetrics:
        """Execute a single epoch of training or evaluation (One-Shot)."""
        if training:
            if optimizer is None:
                msg = "Optimizer required for training mode"
                raise ValueError(msg)
            model.train()
            desc = "Training"
        else:
            model.eval()
            desc = "Evaluating"

        total_loss = torch.tensor(0.0, device=self.device)
        total_ce_loss = torch.tensor(0.0, device=self.device)
        total_degree_loss = torch.tensor(0.0, device=self.device)
        total_crossing_loss = torch.tensor(0.0, device=self.device)
        total_verify_loss = torch.tensor(0.0, device=self.device)
        total_verify_acc = torch.tensor(0.0, device=self.device)
        total_verify_recall_pos = torch.tensor(0.0, device=self.device)
        total_verify_recall_neg = torch.tensor(0.0, device=self.device)
        correct_predictions = torch.tensor(0, device=self.device)
        total_edges = torch.tensor(0, device=self.device)
        perfect_puzzle_stats = []
        num_verify_batches = 0

        loss_weights = self.config["training"].get("loss_weights")
        use_verification = self.config["model"].get("use_verification_head", False)

        context = torch.no_grad() if not training else torch.enable_grad()

        with context:
            if training:
                optimizer.zero_grad()

            for batch_idx, data in enumerate(tqdm(loader, desc=desc, leave=False)):
                data = data.to(self.device)

                # Apply masking logic
                data = self.masking_strategy.apply(
                    data,
                    masking_rate,
                    self.device,
                )

                edge_attr = getattr(data, "edge_attr", None)
                edge_batch = get_edge_batch_indices(data)
                node_type = getattr(data, "node_type", None)

                model_has_verify = (
                    hasattr(model, "use_verification_head")
                    and model.use_verification_head
                )
                should_verify = use_verification and model_has_verify

                if should_verify:
                    logits, verify_logits = model(
                        data.x,
                        data.edge_index,
                        edge_attr=edge_attr,
                        batch=getattr(data, "batch", None),
                        node_type=node_type,
                        return_verification=True,
                    )
                else:
                    logits = model(
                        data.x,
                        data.edge_index,
                        edge_attr=edge_attr,
                        batch=getattr(data, "batch", None),
                        node_type=node_type,
                    )
                    verify_logits = None

                # Use node_type for capacities if available, otherwise fall back
                # to x[:, 0]
                node_capacities = (
                    node_type if node_type is not None else data.x[:, 0].long()
                )
                edge_conflicts = getattr(data, "edge_conflicts", None)

                losses = compute_combined_loss(
                    logits,
                    data.y,
                    data.edge_index,
                    node_capacities,
                    edge_conflicts,
                    data.edge_mask,
                    loss_weights,
                    verify_logits=verify_logits,
                    edge_batch=edge_batch,
                )
                loss = losses["total"]

                total_ce_loss += losses["ce"] * data.num_graphs
                total_degree_loss += losses["degree"] * data.num_graphs
                total_crossing_loss += losses["crossing"] * data.num_graphs
                total_verify_loss += losses["verify"] * data.num_graphs
                total_verify_acc += losses["verify_acc"]
                total_verify_recall_pos += losses["verify_recall_pos"]
                total_verify_recall_neg += losses["verify_recall_neg"]
                if losses["verify"] > 0:
                    num_verify_batches += 1

                if training:
                    scaled_loss = loss / accumulation_steps
                    scaled_loss.backward()
                    if (batch_idx + 1) % accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(loader):
                        optimizer.step()
                        optimizer.zero_grad()

                # Masking for accurate accuracy metrics
                logits_original = logits[data.edge_mask]
                total_loss += loss * data.num_graphs
                pred = logits_original.argmax(dim=-1)
                targets_original = data.y[data.edge_mask]
                correct_predictions += (pred == targets_original).sum()
                total_edges += data.edge_mask.sum()

                edge_batch_original = edge_batch[data.edge_mask]
                # Fix: Pass correct mask for accuracy calculation
                # (filtered original edges)
                accuracy_mask = torch.ones(
                    logits_original.size(0),
                    dtype=torch.bool,
                    device=self.device,
                )
                _, num_perfect, num_total = calculate_batch_perfect_puzzles(
                    logits_original,
                    targets_original,  # Use masked targets here
                    accuracy_mask,
                    edge_batch_original,
                )
                perfect_puzzle_stats.append((num_perfect, num_total))

        num_samples = len(loader.dataset)
        metrics = EpochMetrics()
        metrics.loss = (total_loss / num_samples).item()
        metrics.ce_loss = (total_ce_loss / num_samples).item()
        metrics.degree_loss = (total_degree_loss / num_samples).item()
        metrics.crossing_loss = (total_crossing_loss / num_samples).item()
        metrics.verify_loss = (total_verify_loss / num_samples).item()
        metrics.verify_balanced_acc = (
            (total_verify_acc / num_verify_batches).item()
            if num_verify_batches > 0
            else 0.0
        )
        metrics.verify_recall_pos = (
            (total_verify_recall_pos / num_verify_batches).item()
            if num_verify_batches > 0
            else 0.0
        )
        metrics.verify_recall_neg = (
            (total_verify_recall_neg / num_verify_batches).item()
            if num_verify_batches > 0
            else 0.0
        )
        metrics.accuracy = (correct_predictions / total_edges).item()

        total_perfect = sum(p for p, _ in perfect_puzzle_stats)
        total_puzzles = sum(t for _, t in perfect_puzzle_stats)
        metrics.perfect_accuracy = (
            total_perfect / total_puzzles if total_puzzles > 0 else 0.0
        )

        return metrics
