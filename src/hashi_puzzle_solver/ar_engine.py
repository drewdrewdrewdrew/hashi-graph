"""Auto-Regressive (AR) training engine for Hashi GNN."""

from typing import Any

import torch
import torch.nn.functional as func
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from .ar_utils import get_edge_feature_indices, rewire_component_meta_edges_batch
from .train_utils import update_node_features


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
            self.num_edges, dtype=torch.long, device=device
        )
        self.is_solved = False
        self.is_failed = False
        self.steps_survived = 0
        self.total_steps = 0
        self.mistake_made = False
        self.rollouts_seen = 0
        self.rollouts_aced = 0

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
            self.num_edges, dtype=torch.long, device=self.data.x.device
        )
        self.is_solved = False
        self.is_failed = False
        self.steps_survived = 0
        self.total_steps = 0
        self.mistake_made = False
        # Note: rollouts_seen and rollouts_aced are not reset here
        # so they can be accumulated across an epoch if needed.


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
        self.k = config["training"].get("ar_k", 1)  # Top-K actions per step

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
        Run a single epoch of AR training or evaluation using batch-wise rollout.

        Iterates over the DataLoader. For each batch, performs a full rollout
        until completion or max_steps.

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
        total_steps_loss = 0

        total_precision_accum = 0.0
        total_actions_count = 0

        total_puzzles = 0
        total_aced_puzzles = 0
        total_msuf_accum = 0.0

        # Limit rollout steps per batch to prevent infinite loops
        max_rollout_steps = self.config["training"].get("ar_max_steps", 100)

        desc = "AR Training" if training else "AR Evaluating"

        for batch in tqdm(loader, desc=desc):
            # Move batch to device and split into individual Data objects
            batch = batch.to(self.device)
            data_list = batch.to_data_list()

            # Initialize AR states for this batch
            active_puzzles = [ARState(d, self.device) for d in data_list]
            all_puzzles_in_batch = list(active_puzzles)  # Keep reference for metrics

            step = 0
            while active_puzzles and step < max_rollout_steps:
                # Collate only the active puzzles
                active_list = [p.data for p in active_puzzles]
                collated_data = Batch.from_data_list(active_list)

                # 1. Update node features (unused capacity)
                curr_br = torch.cat([p.current_bridges for p in active_puzzles])
                collated_data.x = update_node_features(
                    collated_data.x,
                    curr_br,
                    collated_data.edge_index,
                    collated_data.node_type,
                    self.config["model"],
                )

                # 2. Update edge features (Current Bridge Count)
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

                # 3. Update component meta edges (Rewire)
                if self.config["model"].get("use_component_meta", False):
                    rewire_component_meta_edges_batch(collated_data, active_puzzles)

                # 4. Model Forward Pass
                edge_attr = getattr(collated_data, "edge_attr", None)
                logits = self.model(
                    collated_data.x,
                    collated_data.edge_index,
                    edge_attr=edge_attr,
                    batch=collated_data.batch,
                    node_type=collated_data.node_type,
                )

                # Puzzle edges only
                edge_mask = collated_data.edge_mask
                puzzle_logits = logits[edge_mask]

                # Ground truth for next best actions (1 if more bridges, 0 otherwise)
                puzzle_targets = (
                    curr_br[edge_mask] < collated_data.y[edge_mask]
                ).long()

                # 5. Loss and Update
                loss = func.cross_entropy(puzzle_logits, puzzle_targets)

                if training and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                total_steps_loss += 1

                # 6. Rollout Logic (Action Selection)
                with torch.no_grad():
                    probs = func.softmax(puzzle_logits, dim=-1)[:, 1]
                    row_idx = collated_data.edge_index[0, edge_mask]
                    edge_batch = collated_data.batch[row_idx]

                    for i, p in enumerate(active_puzzles):
                        p.total_steps += 1

                        # Find edges belonging to this puzzle
                        puzzle_rel_indices = (edge_batch == i).nonzero(
                            as_tuple=True
                        )[0]
                        if len(puzzle_rel_indices) == 0:
                            continue

                        p_probs = probs[puzzle_rel_indices]
                        p_targets = puzzle_targets[puzzle_rel_indices]

                        # Top-K
                        k_val = min(self.k, len(p_probs))
                        top_k_probs, top_k_indices_rel = torch.topk(p_probs, k_val)

                        threshold = self.config["training"].get("ar_threshold", 0.5)
                        mask = top_k_probs > threshold
                        selected_indices_rel = top_k_indices_rel[mask]

                        if len(selected_indices_rel) > 0:
                            correct_actions = (
                                (p_targets[selected_indices_rel] == 1).sum().item()
                            )
                            total_actions = len(selected_indices_rel)

                            total_precision_accum += correct_actions
                            total_actions_count += total_actions

                            if correct_actions < total_actions:
                                p.mistake_made = True

                            if not p.mistake_made:
                                p.steps_survived += 1

                            # Apply valid actions (only correct ones)
                            for idx_rel in selected_indices_rel:
                                if p_targets[idx_rel] == 1:
                                    # Map back to global edge index
                                    p_edge_mask = p.data.edge_mask
                                    p_indices = p_edge_mask.nonzero(as_tuple=True)[0]
                                    global_idx = p_indices[idx_rel]
                                    p.current_bridges[global_idx] += 1

                        # Check completion
                        p_sol_bridges = p.data.y[p.data.edge_mask]
                        p_cur_bridges = p.current_bridges[p.data.edge_mask]
                        solved = torch.all(p_cur_bridges == p_sol_bridges)

                        # If ground truth says no more moves, but we aren't solved,
                        # we are stuck/failed.
                        no_moves = torch.all(p_targets == 0)

                        if solved or no_moves:
                            p.is_solved = True
                            p.rollouts_seen = 1
                            if solved and not p.mistake_made:
                                p.rollouts_aced = 1

                # Filter out completed puzzles
                active_puzzles = [p for p in active_puzzles if not p.is_solved]
                step += 1

                # End of batch processing - Aggregate metrics
            for p in all_puzzles_in_batch:
                total_puzzles += 1
                if p.rollouts_aced == 1:
                    total_aced_puzzles += 1

                # Mean Steps Until Failure (normalized by total steps taken)
                if p.total_steps > 0:
                    total_msuf_accum += p.steps_survived / p.total_steps

        # Epoch Aggregation
        avg_loss = total_loss / total_steps_loss if total_steps_loss > 0 else 0.0
        avg_precision = (
            total_precision_accum / total_actions_count
            if total_actions_count > 0
            else 0.0
        )
        puzzle_aced_rate = (
            total_aced_puzzles / total_puzzles if total_puzzles > 0 else 0.0
        )
        avg_msuf = total_msuf_accum / total_puzzles if total_puzzles > 0 else 0.0

        return {
            "loss": avg_loss,
            "precision": avg_precision,
            "aced_rate": puzzle_aced_rate,
            "avg_rollouts_aced": puzzle_aced_rate,  # Same as puzzle rate
            "puzzle_aced_rate": puzzle_aced_rate,
            "msuf": avg_msuf,
        }
