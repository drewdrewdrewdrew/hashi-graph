"""Auto-Regressive (AR) trainer for Hashi Puzzle Solver."""

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from ..utils.ar_utils import (
    get_edge_feature_indices,
    rewire_hierarchical_edges,
)
from ..utils.common import custom_collate_with_conflicts
from ..utils.train_utils import (
    get_edge_batch_indices,
    update_node_features,
)
from .base import BaseTrainer, EpochMetrics


def redistribute_edge_conflicts(batch: Any, data_list: list[Data]) -> None:
    """Ensure d.edge_conflicts list is populated from d.edge_conflict_index."""
    for d in data_list:
        if hasattr(d, "edge_conflict_index") and d.edge_conflict_index is not None:
            if d.edge_conflict_index.size(1) > 0:
                conflicts = d.edge_conflict_index.t().tolist()
                d.edge_conflicts = [tuple(c) for c in conflicts]
            else:
                d.edge_conflicts = []
        elif not hasattr(d, "edge_conflicts"):
            d.edge_conflicts = []


class ARState:
    """Maintains the state of an active puzzle during AR rollout."""

    def __init__(self, data: Data, device: torch.device):
        self.data = data.clone().to(device)
        self.num_edges = self.data.edge_index.size(1)
        self.num_islands = (self.data.node_type <= 8).sum().item()
        self.current_bridges = torch.zeros(
            self.num_edges, dtype=torch.float, device=device
        )
        self.model_solved = False

    def reset(self, new_data: Data) -> None:
        self.data = new_data.clone().to(self.data.x.device)
        self.num_edges = self.data.edge_index.size(1)
        self.num_islands = (self.data.node_type <= 8).sum().item()
        self.current_bridges = torch.zeros(
            self.num_edges, dtype=torch.float, device=self.data.x.device
        )
        self.model_solved = False


class ARTrainer(BaseTrainer):
    """
    Trainer for Auto-Regressive Hashi solving.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Edge feature indices for AR state updates
        edge_map = get_edge_feature_indices(self.config["model"])
        self.bridge_label_idx = edge_map.get("bridge_label")
        self.is_labeled_idx = edge_map.get("is_labeled")

    def run_epoch(
        self,
        loader: DataLoader,
        training: bool = True,
        epoch: int = 1,
        total_epochs: int = 1,
    ) -> EpochMetrics:
        """Execute a single epoch of AR training or evaluation."""
        if training:
            self.model.train()
            desc = f"AR Training Epoch {epoch}"
        else:
            self.model.eval()
            desc = f"AR Evaluating Epoch {epoch}"

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

        max_rollout_steps = self.config["training"].get("ar_max_steps", 100)
        use_verification = self.config["model"].get("use_verification_head", False)
        gumbel_temperature = self.config["training"].get("gumbel_temperature", 1.0)

        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch.to(self.device)
            data_list = batch.to_data_list()

            # Redistribute edge conflicts for individual states
            redistribute_edge_conflicts(batch, data_list)

            states = [ARState(d, self.device) for d in data_list]

            if training:
                self.optimizer.zero_grad()

            step = 0
            active_mask = torch.ones(len(states), dtype=torch.bool, device=self.device)
            total_rollout_loss = 0.0

            while active_mask.any() and step < max_rollout_steps:
                active_indices = active_mask.nonzero(as_tuple=True)[0]
                active_list = [states[i].data for i in active_indices]
                collated_data = custom_collate_with_conflicts(active_list)
                collated_data = collated_data.to(self.device)

                # Update node and edge features based on current AR state
                curr_br = torch.cat([states[i].current_bridges for i in active_indices])
                collated_data.x = update_node_features(
                    collated_data.x,
                    curr_br,
                    collated_data.edge_index,
                    collated_data.node_type,
                    self.config["model"],
                )

                if self.bridge_label_idx is not None:
                    row, col = collated_data.edge_index
                    is_puzzle_edge = (
                        (collated_data.node_type[row] <= 8)
                        & (collated_data.node_type[row] > 0)
                        & (collated_data.node_type[col] <= 8)
                        & (collated_data.node_type[col] > 0)
                    )
                    bridge_feat = curr_br[is_puzzle_edge].float()
                    collated_data.edge_attr[is_puzzle_edge, self.bridge_label_idx] = bridge_feat
                    if self.is_labeled_idx is not None:
                        collated_data.edge_attr[is_puzzle_edge, self.is_labeled_idx] = 1.0

                if self.config["model"].get("use_component_meta", False):
                    active_puzzles_list = [states[i] for i in active_indices]
                    collated_data = rewire_hierarchical_edges(
                        collated_data, self.config["model"], active_puzzles=active_puzzles_list
                    )

                # Forward pass
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

                # Compute loss
                edge_mask = collated_data.edge_mask
                edge_batch = get_edge_batch_indices(collated_data)
                node_type = getattr(collated_data, "node_type", None)
                node_capacities = node_type if node_type is not None else collated_data.x[:, 0].long()
                edge_conflicts = getattr(collated_data, "edge_conflict_index", None)

                losses = self.loss_calculator(
                    logits=logits,
                    targets=collated_data.y,
                    edge_index=collated_data.edge_index,
                    node_capacities=node_capacities,
                    edge_conflict_index=edge_conflicts,
                    edge_mask=edge_mask,
                    verify_logits=verify_logits,
                    edge_batch=edge_batch,
                )

                loss = losses["total"]
                if training:
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

                # Action selection and state update
                puzzle_logits = logits[edge_mask]
                puzzle_targets = collated_data.y[edge_mask]

                if training:
                    soft_one_hot = F.gumbel_softmax(
                        puzzle_logits, tau=gumbel_temperature, hard=True, dim=-1
                    )
                    model_pred = soft_one_hot[:, 1] + 2 * soft_one_hot[:, 2]
                else:
                    with torch.no_grad():
                        model_pred = puzzle_logits.argmax(dim=-1).float()

                # Accuracy and Teacher Forcing
                with torch.no_grad():
                    pred_count_hard = model_pred.round().long() if training else model_pred.long()
                    acc = (pred_count_hard == puzzle_targets).float().mean().item()
                    total_accuracy_accum += acc
                    total_edges_count += 1

                if training and self.masking_strategy.enabled:
                    ratio = self.masking_strategy.get_rate(epoch, total_epochs)
                    teacher_mask = (torch.rand(model_pred.shape, device=self.device) < ratio)
                    action_to_apply = torch.where(teacher_mask, puzzle_targets.float(), model_pred)
                else:
                    action_to_apply = model_pred

                # Update state for each active puzzle
                row_idx = collated_data.edge_index[0, edge_mask]
                edge_batch_indices = collated_data.batch[row_idx]

                for i, state_idx in enumerate(active_indices):
                    p_rel_indices = (edge_batch_indices == i).nonzero(as_tuple=True)[0]
                    if len(p_rel_indices) == 0:
                        continue

                    p_action = action_to_apply[p_rel_indices]
                    p_edge_mask = states[state_idx].data.edge_mask
                    p_indices = p_edge_mask.nonzero(as_tuple=True)[0]

                    new_bridges = states[state_idx].current_bridges.clone()
                    new_bridges[p_indices] = p_action
                    states[state_idx].current_bridges = new_bridges

                    with torch.no_grad():
                        p_target = puzzle_targets[p_rel_indices]
                        p_pred_hard = model_pred[p_rel_indices].round().long() if training else model_pred[p_rel_indices].long()
                        if torch.all(p_pred_hard == p_target):
                            states[state_idx].model_solved = True
                            active_mask[state_idx] = False

                step += 1

            if training:
                if isinstance(total_rollout_loss, torch.Tensor):
                    total_rollout_loss.backward()
                self.optimizer.step()

            total_solved_puzzles += sum(1 for s in states if s.model_solved)
            total_puzzles += len(states)

        # Epoch aggregation
        metrics = EpochMetrics()
        if total_steps > 0:
            metrics.loss = total_loss / total_steps
            metrics.ce_loss = total_ce_loss / total_steps
            metrics.degree_loss = total_degree_loss / total_steps
            metrics.crossing_loss = total_crossing_loss / total_steps
            metrics.verify_loss = total_verify_loss / total_steps

            if num_verify_batches > 0:
                metrics.verify_balanced_acc = total_verify_acc / num_verify_batches
                metrics.verify_recall_pos = total_verify_recall_pos / num_verify_batches
                metrics.verify_recall_neg = total_verify_recall_neg / num_verify_batches

        metrics.accuracy = total_accuracy_accum / total_edges_count if total_edges_count > 0 else 0.0
        metrics.perfect_accuracy = total_solved_puzzles / total_puzzles if total_puzzles > 0 else 0.0

        return metrics
