"""Auto-Regressive (AR) training engine for Hashi GNN."""

import bisect
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from .ar_utils import get_edge_feature_indices, rewire_component_meta_edges_batch
from .losses import compute_combined_loss
from .train_utils import (
    calculate_perfect_puzzle_accuracy,
    get_edge_batch_indices,
    update_node_features,
)
from .utils import custom_collate_with_conflicts


def redistribute_edge_conflicts(batch: Any, data_list: list[Data]) -> None:
    """
    Manually redistribute edge conflicts from batch to individual Data objects.

    Batch.to_data_list() loses custom attributes like edge_conflicts.
    We recover them using edge offsets.
    """
    if not hasattr(batch, "edge_conflicts") or not batch.edge_conflicts:
        return

    # Calculate edge offsets
    edge_counts = [d.edge_index.size(1) for d in data_list]
    edge_offsets = [0]
    for count in edge_counts:
        edge_offsets.append(edge_offsets[-1] + count)

    # Initialize empty lists
    for d in data_list:
        d.edge_conflicts = []

    # Distribute conflicts
    # batch.edge_conflicts is a list of [e1, e2]
    for e1, e2 in batch.edge_conflicts:
        # Find which graph this edge belongs to
        # edge_offsets[i] <= e1 < edge_offsets[i+1]
        graph_idx = bisect.bisect_right(edge_offsets, e1) - 1
        
        if graph_idx < 0 or graph_idx >= len(data_list):
            continue

        offset = edge_offsets[graph_idx]
        local_e1 = e1 - offset
        local_e2 = e2 - offset
        
        data_list[graph_idx].edge_conflicts.append((local_e1, local_e2))


class ARState:
    """Maintains the state of an active puzzle during AR rollout."""

    def __init__(self, data: Data, device: torch.device):
        """
        Initialize AR state.

        Args:
            data: Initial puzzle data.
            device: Compute device.
        """
        self.data = data.clone().to(device)
        self.num_edges = self.data.edge_index.size(1)
        self.num_islands = (self.data.node_type <= 8).sum().item()
        self.current_bridges = torch.zeros(
            self.num_edges, dtype=torch.float, device=device
        )

    def reset(self, new_data: Data) -> None:
        """
        Reset state with new data.

        Args:
            new_data: Fresh puzzle data.
        """
        self.data = new_data.clone().to(self.data.x.device)
        self.num_edges = self.data.edge_index.size(1)
        self.num_islands = (self.data.node_type <= 8).sum().item()
        self.current_bridges = torch.zeros(
            self.num_edges, dtype=torch.float, device=self.data.x.device
        )


class ARTrainer:
    """Trainer for Auto-Regressive Hashi solving."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        """
        Initialize AR trainer.

        Args:
            model: GNN model.
            config: Training configuration.
            device: Compute device.
        """
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
        optimizer: Optimizer | None = None,
        training: bool = True,
    ) -> dict[str, float]:
        """
        Run a single epoch of AR training or evaluation using the Unified Loop.

        Implements Gradient Accumulation (1 update per batch), Ordinal Action Space,
        and state overwriting.

        Args:
            loader: Standard DataLoader.
            optimizer: Optimizer for training.
            training: Whether to perform backprop.

        Returns
        -------
            Dictionary of metrics.
        """
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
        total_verify_recall_pos = 0.0
        total_verify_recall_neg = 0.0
        total_steps = 0
        total_accuracy_accum = 0.0
        total_edges_count = 0
        num_verify_batches = 0

        total_puzzles = 0
        total_solved_puzzles = 0

        # Limit rollout steps per batch
        max_rollout_steps = self.config["training"].get("ar_max_steps", 100)
        loss_weights = self.config["training"].get("loss_weights")
        use_verification = self.config["model"].get("use_verification_head", False)
        gumbel_temperature = self.config["training"].get("gumbel_temperature", 1.0)

        desc = "AR Training" if training else "AR Evaluating"

        for batch in tqdm(loader, desc=desc, leave=False):
            # Move batch to device (to get the initial data on device)
            batch = batch.to(self.device)
            data_list = batch.to_data_list()
            
            # Fix: Redistribute edge_conflicts lost by to_data_list()
            redistribute_edge_conflicts(batch, data_list)

            # Initialize AR states for this batch
            states = [ARState(d, self.device) for d in data_list]

            if training and optimizer is not None:
                optimizer.zero_grad()

            step = 0
            # Track which puzzles are still active (not solved)
            active_mask = torch.ones(len(states), dtype=torch.bool, device=self.device)
            total_rollout_loss = 0.0

            while active_mask.any() and step < max_rollout_steps:
                # 1. Prepare data for active puzzles
                active_indices = active_mask.nonzero(as_tuple=True)[0]
                active_list = [states[i].data for i in active_indices]
                collated_data = custom_collate_with_conflicts(active_list)
                collated_data = collated_data.to(self.device)

                # 2. Update node features (unused capacity)
                curr_br = torch.cat(
                    [states[i].current_bridges for i in active_indices]
                )
                collated_data.x = update_node_features(
                    collated_data.x,
                    curr_br,
                    collated_data.edge_index,
                    collated_data.node_type,
                    self.config["model"],
                )

                # 3. Update edge features (Current Bridge Count)
                if self.bridge_label_idx is not None:
                    row, col = collated_data.edge_index
                    is_puzzle_edge = (
                        (collated_data.node_type[row] <= 8)
                        & (collated_data.node_type[row] > 0)
                        & (collated_data.node_type[col] <= 8)
                        & (collated_data.node_type[col] > 0)
                    )
                    bridge_feat = curr_br[is_puzzle_edge].float()
                    collated_data.edge_attr[
                        is_puzzle_edge, self.bridge_label_idx
                    ] = bridge_feat
                    if self.is_labeled_idx is not None:
                        collated_data.edge_attr[
                            is_puzzle_edge, self.is_labeled_idx
                        ] = 1.0

                # 4. Update component meta edges (Rewire)
                if self.config["model"].get("use_component_meta", False):
                    # Filter active puzzles for rewire helper
                    active_puzzles_list = [states[i] for i in active_indices]
                    rewire_component_meta_edges_batch(
                        collated_data, active_puzzles_list
                    )

                # 5. Model Forward Pass
                edge_attr = getattr(collated_data, "edge_attr", None)

                model_has_verify = (
                    hasattr(self.model, "use_verification_head")
                    and self.model.use_verification_head
                )
                should_verify = use_verification and model_has_verify

                if should_verify:
                    logits, verify_logits = self.model(
                        collated_data.x,
                        collated_data.edge_index,
                        edge_attr=edge_attr,
                        batch=collated_data.batch,
                        node_type=collated_data.node_type,
                        return_verification=True,
                    )
                else:
                    logits = self.model(
                        collated_data.x,
                        collated_data.edge_index,
                        edge_attr=edge_attr,
                        batch=collated_data.batch,
                        node_type=collated_data.node_type,
                    )
                    verify_logits = None

                # 6. Combined Loss (CE + Degree + Crossing)
                edge_mask = collated_data.edge_mask
                edge_batch = get_edge_batch_indices(collated_data)
                node_type = getattr(collated_data, "node_type", None)
                node_capacities = (
                    node_type if node_type is not None else collated_data.x[:, 0].long()
                )
                edge_conflicts = getattr(collated_data, "edge_conflicts", None)

                losses = compute_combined_loss(
                    logits,
                    collated_data.y,
                    collated_data.edge_index,
                    node_capacities,
                    edge_conflicts,
                    edge_mask,
                    loss_weights,
                    verify_logits=verify_logits,
                    edge_batch=edge_batch,
                )
                loss = losses["total"]

                if training and optimizer is not None:
                    # Accumulate loss over the rollout steps (BPTT)
                    # We sum the loss across steps to maintain strong signal
                    total_rollout_loss = total_rollout_loss + loss

                total_loss += loss.item()
                total_ce_loss += losses["ce"].item()
                total_degree_loss += losses["degree"].item()
                total_crossing_loss += losses["crossing"].item()
                total_verify_loss += losses["verify"].item()
                total_verify_acc += losses["verify_acc"].item()
                total_verify_recall_pos += losses["verify_recall_pos"].item()
                total_verify_recall_neg += losses["verify_recall_neg"].item()
                if losses["verify"] > 0:
                    num_verify_batches += 1
                total_steps += 1

                # 7. Action Selection (State Update)
                puzzle_logits = logits[edge_mask]
                puzzle_targets = collated_data.y[edge_mask]

                if training:
                    # Gumbel Softmax for differentiable action selection
                    soft_one_hot = F.gumbel_softmax(
                        puzzle_logits, tau=gumbel_temperature, hard=True, dim=-1
                    )
                    # Expected bridge count: sum(prob * value) -> 0*p0 + 1*p1 + 2*p2
                    pred_count = soft_one_hot[:, 1] + 2 * soft_one_hot[:, 2]
                else:
                    with torch.no_grad():
                        pred_count = puzzle_logits.argmax(dim=-1).float()

                # Accuracy metric (Edge-wise) - use detached/hard values for metrics
                with torch.no_grad():
                    # For metric calculation, round the soft predictions to nearest integer
                    pred_count_hard = (
                        pred_count.round().long() if training else pred_count.long()
                    )
                    total_accuracy_accum += (
                        (pred_count_hard == puzzle_targets).float().mean().item()
                    )
                    total_edges_count += 1

                # Update current_bridges for each state
                row_idx = collated_data.edge_index[0, edge_mask]
                edge_batch = collated_data.batch[row_idx]

                for i, state_idx in enumerate(active_indices):
                    p_rel_indices = (edge_batch == i).nonzero(as_tuple=True)[0]
                    if len(p_rel_indices) == 0:
                        continue

                    # Get predictions for this puzzle
                    p_pred = pred_count[p_rel_indices]

                    # Update state (clone to avoid in-place modification for BPTT)
                    p_edge_mask = states[state_idx].data.edge_mask
                    p_indices = p_edge_mask.nonzero(as_tuple=True)[0]

                    new_bridges = states[state_idx].current_bridges.clone()
                    new_bridges[p_indices] = p_pred
                    states[state_idx].current_bridges = new_bridges

                    # Check if solved (using hard predictions for logic)
                    with torch.no_grad():
                        p_target = puzzle_targets[p_rel_indices]
                        p_pred_hard = (
                            p_pred.round().long() if training else p_pred.long()
                        )
                        if torch.all(p_pred_hard == p_target):
                            active_mask[state_idx] = False

                step += 1

            if training and optimizer is not None:
                if isinstance(total_rollout_loss, torch.Tensor):
                    total_rollout_loss.backward()
                optimizer.step()

            # End of batch processing - Final Solve Rate
            # Aggregate state for consistent metric calculation
            all_preds = []
            all_targets = []
            all_masks = []
            all_batch = []

            for i, state in enumerate(states):
                all_preds.append(state.current_bridges.detach().round().long())
                all_targets.append(state.data.y)
                all_masks.append(state.data.edge_mask)
                all_batch.append(
                    torch.full(
                        (state.num_edges,),
                        i,
                        device=self.device,
                        dtype=torch.long,
                    )
                )

            solved_rate, n_solved, n_total = calculate_perfect_puzzle_accuracy(
                torch.cat(all_preds),
                torch.cat(all_targets),
                torch.cat(all_masks),
                torch.cat(all_batch),
            )
            total_solved_puzzles += n_solved
            total_puzzles += n_total

        # Epoch Aggregation
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        avg_ce_loss = total_ce_loss / total_steps if total_steps > 0 else 0.0
        avg_degree_loss = total_degree_loss / total_steps if total_steps > 0 else 0.0
        avg_crossing_loss = (
            total_crossing_loss / total_steps if total_steps > 0 else 0.0
        )
        avg_accuracy = (
            total_accuracy_accum / total_edges_count
            if total_edges_count > 0 else 0.0
        )
        solved_rate = (
            total_solved_puzzles / total_puzzles if total_puzzles > 0 else 0.0
        )

        return {
            "loss": avg_loss,
            "ce_loss": avg_ce_loss,
            "degree_loss": avg_degree_loss,
            "crossing_loss": avg_crossing_loss,
            "verify_loss": (
                total_verify_loss / total_steps if total_steps > 0 else 0.0
            ),
            "verify_balanced_acc": (
                total_verify_acc / num_verify_batches
                if num_verify_batches > 0 else 0.0
            ),
            "verify_recall_pos": (
                total_verify_recall_pos / num_verify_batches
                if num_verify_batches > 0 else 0.0
            ),
            "verify_recall_neg": (
                total_verify_recall_neg / num_verify_batches
                if num_verify_batches > 0 else 0.0
            ),
            "accuracy": avg_accuracy,
            "perfect_accuracy": solved_rate,
        }
