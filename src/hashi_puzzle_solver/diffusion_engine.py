"""Diffusion training engine for Hashi GNN."""

from typing import Any
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ar_utils import get_edge_feature_indices
from .diffusion_utils import inject_noise
from .losses import compute_combined_loss
from .train_utils import (
    get_edge_batch_indices,
    calculate_batch_perfect_puzzles,
    update_node_features,
)
from .utils import custom_collate_with_conflicts

class DiffusionTrainer:
    """Trainer for Denoising Diffusion Hashi solving."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

        # Edge feature indices
        edge_map = get_edge_feature_indices(config["model"])
        self.bridge_label_idx = edge_map.get("bridge_label")
        self.is_labeled_idx = edge_map.get("is_labeled")

    def run_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
        optimizer: Optimizer | None = None,
        training: bool = True,
        noise_rate: float = 0.0,
    ) -> dict[str, float]:
        """Run a single epoch of Diffusion training or evaluation."""
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_degree_loss = 0.0
        total_crossing_loss = 0.0
        total_verify_loss = 0.0
        total_verify_acc = 0.0
        total_steps = 0
        total_accuracy_accum = 0.0
        total_edges_count = 0
        total_puzzles = 0
        total_solved_puzzles = 0

        loss_weights = self.config["training"].get("loss_weights")
        use_verification = self.config["model"].get("use_verification_head", False)

        desc = "Diffusion Training" if training else "Diffusion Evaluating"

        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch.to(self.device)
            
            # 1. Inject Noise
            # For training, we inject noise into the ground truth.
            # For evaluation, we might want to do full rollout, but Phase 1 says 
            # "Dual-Path Validation": 
            # - Standard Validation: same noise/corruption as training.
            # - Iterative Rollout Validation: full loop (done every N epochs).
            # For now, we implement the Standard Validation path.
            
            data = inject_noise(
                batch,
                noise_rate,
                self.bridge_label_idx,
                self.is_labeled_idx,
                self.config["model"],
                self.device
            )

            if training and optimizer is not None:
                optimizer.zero_grad()

            # 2. Forward Pass
            edge_attr = getattr(data, "edge_attr", None)
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
                    batch=data.batch,
                    node_type=data.node_type,
                    return_verification=True,
                )
            else:
                logits = self.model(
                    data.x,
                    data.edge_index,
                    edge_attr=edge_attr,
                    batch=data.batch,
                    node_type=data.node_type,
                )
                verify_logits = None

            # 3. Loss Calculation
            edge_mask = data.edge_mask
            edge_batch = get_edge_batch_indices(data)
            node_type = getattr(data, "node_type", None)
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
                edge_mask,
                loss_weights,
                verify_logits=verify_logits,
                edge_batch=edge_batch,
            )
            loss = losses["total"]

            if training and optimizer is not None:
                loss.backward()
                optimizer.step()

            # 4. Metrics
            total_loss += loss.item()
            total_ce_loss += losses["ce"].item()
            total_degree_loss += losses["degree"].item()
            total_crossing_loss += losses["crossing"].item()
            total_verify_loss += losses["verify"].item()
            total_verify_acc += losses["verify_acc"].item()
            total_steps += 1

            # Edge-wise accuracy
            puzzle_logits = logits[edge_mask]
            puzzle_targets = data.y[edge_mask]
            with torch.no_grad():
                pred = puzzle_logits.argmax(dim=-1)
                acc = (pred == puzzle_targets).float().mean().item()
                total_accuracy_accum += acc
                total_edges_count += 1
                
                # Perfect puzzle accuracy
                _, num_perfect, num_total = calculate_batch_perfect_puzzles(
                    puzzle_logits,
                    puzzle_targets,
                    torch.ones(len(puzzle_targets), dtype=torch.bool, device=self.device),
                    edge_batch[edge_mask]
                )
                total_solved_puzzles += num_perfect
                total_puzzles += num_total

        return {
            "loss": total_loss / total_steps if total_steps > 0 else 0.0,
            "ce_loss": total_ce_loss / total_steps if total_steps > 0 else 0.0,
            "degree_loss": total_degree_loss / total_steps if total_steps > 0 else 0.0,
            "crossing_loss": total_crossing_loss / total_steps if total_steps > 0 else 0.0,
            "verify_loss": total_verify_loss / total_steps if total_steps > 0 else 0.0,
            "accuracy": total_accuracy_accum / total_edges_count if total_edges_count > 0 else 0.0,
            "perfect_accuracy": total_solved_puzzles / total_puzzles if total_puzzles > 0 else 0.0,
        }

    def run_rollout(
        self,
        loader: DataLoader,
        max_steps: int = 20,
        checkpoints: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Perform iterative cleanup (inference) on a batch of puzzles.
        Initial state is an empty board.
        
        Args:
            loader: DataLoader for validation/test set.
            max_steps: Maximum number of iterative steps.
            checkpoints: Steps at which to record perfect puzzle accuracy.
            
        Returns:
            dict: Metrics including accuracy and perfect puzzle accuracy at checkpoints.
        """
        if checkpoints is None:
            checkpoints = [1, 3, 5, 10, 20]
            
        self.model.eval()
        
        total_puzzles = 0
        puzzle_solved_at_k = {k: 0 for k in checkpoints}
        final_accuracy_accum = 0.0
        total_batches = 0
        
        for batch in tqdm(loader, desc="Diffusion Rollout", leave=False):
            batch = batch.to(self.device)
            num_graphs = batch.num_graphs
            total_puzzles += num_graphs
            total_batches += 1
            
            # Start with empty board
            current_bridges = torch.zeros(
                batch.edge_index.size(1), device=self.device
            ).float()

            # Randomize ONLY the puzzle edges (where edge_mask is True)
            # This matches inject_noise behavior at 100% noise
            if hasattr(batch, "edge_mask") and batch.edge_mask is not None:
                mask = batch.edge_mask
                num_masked = mask.sum()
                if num_masked > 0:
                    current_bridges[mask] = torch.randint(
                        0, 3, (num_masked,), device=self.device
                    ).float()
            else:
                # Fallback if no mask exists
                current_bridges = torch.randint(
                    0, 3, (batch.edge_index.size(1),), device=self.device
                ).float()
            
            # Track solved status per puzzle in batch
            edge_batch = get_edge_batch_indices(batch)
            puzzle_solved = torch.zeros(num_graphs, dtype=torch.bool, device=self.device)
            
            # Working copy of data
            data = batch.clone()
            
            for step_idx in range(1, max_steps + 1):
                # Update features for current state
                if self.bridge_label_idx is not None:
                    data.edge_attr[:, self.bridge_label_idx] = current_bridges
                    data.edge_attr[:, self.is_labeled_idx] = 1.0
                
                if self.config["model"].get("use_unused_capacity", True):
                    data.x = update_node_features(
                        batch.x, # Original x with full capacities
                        current_bridges,
                        data.edge_index,
                        data.node_type,
                        self.config["model"]
                    )
                
                with torch.no_grad():
                    logits = self.model(
                        data.x,
                        data.edge_index,
                        edge_attr=data.edge_attr,
                        batch=data.batch,
                        node_type=data.node_type
                    )
                    
                    # Update board state
                    pred = logits.argmax(dim=-1).float()
                    # For Phase 1, we use "Greedy" update: replace with prediction
                    current_bridges = pred
                    
                    # Check which puzzles are solved
                    edge_mask = data.edge_mask
                    puzzle_targets = data.y[edge_mask]
                    puzzle_preds = current_bridges[edge_mask]
                    
                    # Check per puzzle
                    for i in range(num_graphs):
                        if puzzle_solved[i]:
                            continue
                        
                        mask_i = (edge_batch[edge_mask] == i)
                        if torch.all(puzzle_preds[mask_i] == puzzle_targets[mask_i]):
                            puzzle_solved[i] = True
                            
                # Record checkpoints
                if step_idx in checkpoints:
                    puzzle_solved_at_k[step_idx] += puzzle_solved.sum().item()
                
                if puzzle_solved.all():
                    # If all solved, we still need to fill remaining checkpoints
                    for k in checkpoints:
                        if k > step_idx:
                            puzzle_solved_at_k[k] += puzzle_solved.sum().item()
                    break
            
            # Final accuracy for this batch
            edge_mask = data.edge_mask
            puzzle_targets = data.y[edge_mask]
            puzzle_preds = current_bridges[edge_mask]
            final_accuracy_accum += (puzzle_preds == puzzle_targets).float().mean().item()
            
        # Aggregate results
        results = {
            f"perfect_acc_k{k}": puzzle_solved_at_k[k] / total_puzzles if total_puzzles > 0 else 0.0
            for k in checkpoints
        }
        results["accuracy"] = final_accuracy_accum / total_batches if total_batches > 0 else 0.0
        # Also include the last checkpoint as the general perfect_accuracy
        if checkpoints:
            results["perfect_accuracy"] = results[f"perfect_acc_k{checkpoints[-1]}"]
            
        return results
