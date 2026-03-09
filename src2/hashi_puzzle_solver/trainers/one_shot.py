"""One-shot trainer for Hashi Puzzle Solver."""

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .base import BaseTrainer, EpochMetrics
from ..utils.train_utils import calculate_batch_perfect_puzzles, get_edge_batch_indices


class OneShotTrainer(BaseTrainer):
    """
    Standard one-shot trainer that predicts all bridge counts in a single forward pass.
    """

    def run_epoch(
        self,
        loader: DataLoader,
        training: bool = True,
        epoch: int = 1,
        total_epochs: int = 1,
    ) -> EpochMetrics:
        """Execute a single epoch of training or evaluation."""
        if training:
            self.model.train()
            desc = f"Training Epoch {epoch}"
        else:
            self.model.eval()
            desc = f"Evaluating Epoch {epoch}"

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

        accumulation_steps = self.config["training"].get("accumulation_steps", 1)
        use_verification = self.config["model"].get("use_verification_head", False)

        context = torch.no_grad() if not training else torch.enable_grad()

        with context:
            if training:
                self.optimizer.zero_grad()

            for batch_idx, data in enumerate(tqdm(loader, desc=desc, leave=False)):
                data = data.to(self.device)

                # Apply masking logic
                data = self.masking_strategy.apply(
                    data,
                    self.current_masking_rate,
                    self.device,
                )

                edge_attr = getattr(data, "edge_attr", None)
                edge_batch = get_edge_batch_indices(data)
                node_type = getattr(data, "node_type", None)

                model_has_verify = (
                    hasattr(self.model, "use_verification_head")
                    and self.model.use_verification_head
                )
                should_verify = use_verification and model_has_verify

                if should_verify:
                    logits, verify_logits = self.model(
                        data.x,
                        data.edge_index,
                        edge_attr=edge_attr,
                        edge_type=getattr(data, "edge_type", None),
                        batch=getattr(data, "batch", None),
                        node_type=node_type,
                        return_verification=True,
                    )
                else:
                    logits = self.model(
                        data.x,
                        data.edge_index,
                        edge_attr=edge_attr,
                        edge_type=getattr(data, "edge_type", None),
                        batch=getattr(data, "batch", None),
                        node_type=node_type,
                    )
                    verify_logits = None

                # Use node_type for capacities if available
                node_capacities = (
                    node_type if node_type is not None else data.x[:, 0].long()
                )
                edge_conflicts = getattr(data, "edge_conflict_index", None)

                losses = self.loss_calculator(
                    logits=logits,
                    targets=data.y,
                    edge_index=data.edge_index,
                    node_capacities=node_capacities,
                    edge_conflict_index=edge_conflicts,
                    edge_mask=data.edge_mask,
                    verify_logits=verify_logits,
                    edge_batch=edge_batch,
                )
                
                loss = losses["total"]

                if training:
                    scaled_loss = loss / accumulation_steps
                    scaled_loss.backward()
                    if (batch_idx + 1) % accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(loader):
                        self._optimizer_step()
                        self.optimizer.zero_grad()

                # Update metrics
                num_graphs = data.num_graphs
                total_loss += loss * num_graphs
                total_ce_loss += losses["ce"] * num_graphs
                total_degree_loss += losses["degree"] * num_graphs
                total_crossing_loss += losses["crossing"] * num_graphs
                total_verify_loss += losses["verify"] * num_graphs
                total_verify_acc += losses["verify_acc"]
                total_verify_recall_pos += losses["verify_recall_pos"]
                total_verify_recall_neg += losses["verify_recall_neg"]
                
                if losses["verify"] > 0:
                    num_verify_batches += 1

                # Accuracy metrics
                logits_original = logits[data.edge_mask]
                pred = logits_original.argmax(dim=-1)
                targets_original = data.y[data.edge_mask]
                correct_predictions += (pred == targets_original).sum()
                total_edges += data.edge_mask.sum()

                edge_batch_original = edge_batch[data.edge_mask]
                accuracy_mask = torch.ones(
                    logits_original.size(0),
                    dtype=torch.bool,
                    device=self.device,
                )
                _, num_perfect, num_total = calculate_batch_perfect_puzzles(
                    logits_original,
                    targets_original,
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
        
        if num_verify_batches > 0:
            metrics.verify_balanced_acc = (total_verify_acc / num_verify_batches).item()
            metrics.verify_recall_pos = (total_verify_recall_pos / num_verify_batches).item()
            metrics.verify_recall_neg = (total_verify_recall_neg / num_verify_batches).item()
            
        metrics.accuracy = (correct_predictions / total_edges).item() if total_edges > 0 else 0.0

        total_perfect = sum(p for p, _ in perfect_puzzle_stats)
        total_puzzles = sum(t for _, t in perfect_puzzle_stats)
        metrics.perfect_accuracy = (
            total_perfect / total_puzzles if total_puzzles > 0 else 0.0
        )

        return metrics
