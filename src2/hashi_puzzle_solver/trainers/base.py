"""Base trainer class for Hashi Puzzle Solver."""

import torch
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models.config import HashiModelConfig
from ..models.factory import ModelFactory
from ..losses.calculator import HashiLossCalculator
from ..masking import MaskingStrategy
from ..data import HashiDataset, HashiDatasetCache
from ..utils.common import custom_collate_with_conflicts
from ..utils.train_utils import ModelEMA


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

    def to_tuple(self) -> tuple[float, ...]:
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


class BaseTrainer:
    """
    Base trainer providing shared setup, data loading, and model management logic.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        callbacks: list[Any] | None = None,
    ):
        self.config = config
        self.model_config = HashiModelConfig.from_dict(config)
        self.device = device
        self.callbacks = callbacks or []
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.ema: ModelEMA | None = None
        self.train_loader = None
        self.val_loader = None
        
        self.loss_calculator = HashiLossCalculator(self.model_config)
        self.masking_strategy = MaskingStrategy(config)
        
        self.current_masking_rate = 0.0
        self.best_monitored_value = None

    def _setup(self, train_transform: Any | None = None) -> None:
        """Initialize model, optimizer, scheduler, EMA, and data loaders."""
        training_cfg = self.config["training"]
        self.model = ModelFactory.create_model(self.model_config, self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_cfg["learning_rate"],
            eps=float(training_cfg.get("adam_epsilon", 1e-8)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
        )

        sched_type = training_cfg.get("lr_scheduler", "none")
        if sched_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_cfg["epochs"],
                eta_min=float(training_cfg.get("lr_min", 1e-6)),
            )
        elif sched_type == "plateau":
            early_cfg = training_cfg.get("early_stopping", {})
            plateau_mode = "max" if "acc" in early_cfg.get("monitor", "loss") else "min"
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=plateau_mode,
                factor=float(training_cfg.get("lr_plateau_factor", 0.5)),
                patience=int(training_cfg.get("lr_plateau_patience", 5)),
                min_lr=float(training_cfg.get("lr_min", 1e-6)),
            )

        if training_cfg.get("ema_enabled", False):
            self.ema = ModelEMA(
                self.model,
                decay=float(training_cfg.get("ema_decay", 0.999)),
            )

        self.train_loader = self.create_dataloader(
            split="train",
            transform=train_transform,
        )
        self.val_loader = self.create_dataloader(split="val")

    def _optimizer_step(self) -> None:
        """Clip gradients (if configured), step the optimizer, update EMA."""
        grad_clip = self.config["training"].get("grad_clip_norm")
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)

    def save_model(self, path: str | Path) -> None:
        """Save current model state dict."""
        torch.save(self.model.state_dict(), str(path))

    def load_model(self, path: str | Path) -> None:
        """Load model state dict."""
        self.model.load_state_dict(torch.load(str(path), map_location=self.device))

    def create_dataloader(
        self,
        split: str,
        transform: Any | None = None,
        use_cache: bool = False,
    ) -> DataLoader:
        """Shared dataloader creation logic."""
        data_config = self.config["data"]
        model_config = self.config["model"]
        training_config = self.config["training"]

        legacy_limit = data_config.get("limit")
        split_limit = (
            data_config.get("train_limit") if split == "train"
            else data_config.get("val_limit")
        )
        limit = split_limit if split_limit is not None else legacy_limit

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
                limit=None,
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
                use_potential_crossing=model_config.get(
                    "use_potential_crossing", False
                ),
                use_component_meta=model_config.get("use_component_meta", False),
                use_continuous_edge_labels=model_config.get(
                    "use_continuous_edge_labels", False
                ),
                use_categorical_edge_types=model_config.get(
                    "use_categorical_edge_types", False
                ),
                transform=transform,
            )

        sampler = None
        shuffle = (split == "train")

        if limit is not None:
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
                from torch.utils.data import SubsetRandomSampler
                val_sampler_seed = int(data_config.get("val_sampler_seed", 42))
                generator = torch.Generator().manual_seed(val_sampler_seed)
                indices = torch.randperm(
                    len(dataset),
                    generator=generator,
                )[:num_samples].tolist()
                sampler = SubsetRandomSampler(indices)
                shuffle = False

        return DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=training_config.get("num_workers", 0),
            collate_fn=custom_collate_with_conflicts,
            persistent_workers=training_config.get("use_persistent_workers", False),
            pin_memory=self.device.type == "cuda",
        )

    def run_epoch(self, loader: DataLoader, training: bool = True, epoch: int = 1, total_epochs: int = 1) -> EpochMetrics | dict[str, Any]:
        """To be implemented by subclasses."""
        raise NotImplementedError

    def train(self, train_transform: Any | None = None) -> None:
        """Run main training loop."""
        self._setup(train_transform)

        epochs = self.config["training"]["epochs"]
        eval_interval = self.config["training"].get("eval_interval", 1)
        early_stopping_config = self.config["training"].get("early_stopping", {})

        monitor = early_stopping_config.get("monitor", "loss")
        metric_key = monitor.replace("val_", "")
        if metric_key == "perfect_acc":
            metric_key = "perfect_accuracy"

        stop_mode = "max" if "acc" in metric_key.lower() else "min"
        self.best_monitored_value = float("inf") if stop_mode == "min" else float("-inf")

        early_stopper = EarlyStopper(
            monitor=monitor,
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 0.0),
            mode=stop_mode,
        )

        for callback in self.callbacks:
            callback.on_train_start(self)

        try:
            for epoch in range(1, epochs + 1):
                for callback in self.callbacks:
                    callback.on_epoch_start(self, epoch)

                self.current_masking_rate = self.masking_strategy.get_rate(epoch, epochs)
                
                # Run Training Epoch
                train_results = self.run_epoch(
                    self.train_loader, 
                    training=True, 
                    epoch=epoch, 
                    total_epochs=epochs
                )
                
                if isinstance(train_results, dict):
                    train_metrics = self._dict_to_metrics(train_results)
                else:
                    train_metrics = train_results

                # Memory management
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()

                val_metrics = None
                full_rollout_metrics = None
                
                # Run Validation Epoch (with EMA weights if enabled)
                if epoch % eval_interval == 0:
                    if self.ema is not None:
                        self.ema.apply_shadow(self.model)

                    val_results = self.run_epoch(
                        self.val_loader, 
                        training=False, 
                        epoch=epoch, 
                        total_epochs=epochs
                    )
                    
                    if isinstance(val_results, dict):
                        val_metrics = self._dict_to_metrics(val_results)
                    else:
                        val_metrics = val_results
                    
                    # Optional Rollout for Diffusion/AR
                    if hasattr(self, "run_rollout"):
                        training_cfg_inner = self.config["training"]
                        eval_rollout_interval = training_cfg_inner.get("eval_rollout_interval", 0)
                        if eval_rollout_interval > 0 and epoch % eval_rollout_interval == 0:
                            max_steps = training_cfg_inner.get("diffusion_max_steps", 20)
                            full_rollout_metrics = self.run_rollout(self.val_loader, max_steps=max_steps)

                    # Memory management
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                    elif self.device.type == "cuda":
                        torch.cuda.empty_cache()

                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch, train_metrics, val_metrics, full_rollout_metrics)

                if val_metrics:
                    current_val = getattr(val_metrics, metric_key)

                    # Save best model (EMA weights still active if enabled)
                    is_better = False
                    if stop_mode == "min":
                        if current_val < (self.best_monitored_value - early_stopper.min_delta):
                            is_better = True
                    else:
                        if current_val > (self.best_monitored_value + early_stopper.min_delta):
                            is_better = True

                    if is_better:
                        self.best_monitored_value = current_val
                        model_dir = Path(self.config["training"].get("model_dir", "models"))
                        for cb in self.callbacks:
                            if hasattr(cb, "model_dir"):
                                model_dir = cb.model_dir
                                break
                        self.save_model(model_dir / "model_best.pt")
                        print(f"New best {monitor}: {current_val:.4f}. Model saved.")

                    # Restore training weights after val + save
                    if self.ema is not None:
                        self.ema.restore(self.model)

                    # Step LR scheduler
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_val)
                    elif self.scheduler is not None:
                        self.scheduler.step()

                    if early_stopper.early_stop(current_val):
                        print(f"Early stopping at epoch {epoch}")
                        break
                elif self.scheduler is not None and not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()
        finally:
            for callback in self.callbacks:
                callback.on_train_end(self)

    def _dict_to_metrics(self, results: dict[str, Any]) -> EpochMetrics:
        """Helper to convert result dict to EpochMetrics object."""
        metrics = EpochMetrics()
        for key, value in results.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        return metrics
