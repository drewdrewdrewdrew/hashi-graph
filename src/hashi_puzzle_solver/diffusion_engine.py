"""Diffusion training engine for Hashi GNN."""

from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ar_utils import get_edge_feature_indices
from .diffusion_utils import (
    estimate_signal_noise_stats,
    inject_continuous_noise,
    inject_flow_noise,
    inject_noise,
)
from .losses import compute_combined_loss
from .train_utils import (
    calculate_batch_perfect_puzzles,
    get_edge_batch_indices,
    update_node_features,
)


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
        self.bridge_logits_idx = edge_map.get("bridge_logits")

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
        total_noise_loss = 0.0
        total_verify_acc = 0.0
        total_verify_recall_pos = 0.0
        total_verify_recall_neg = 0.0
        total_steps = 0
        total_accuracy_accum = 0.0
        total_edges_count = 0
        total_puzzles = 0
        total_solved_puzzles = 0
        num_verify_batches = 0

        training_cfg = self.config["training"]
        mode = training_cfg.get("mode", "diff-discrete").lower()
        loss_weights = training_cfg.get("loss_weights")
        use_verification = self.config["model"].get("use_verification_head", False)
        use_noise_head = self.config["model"].get("use_noise_head", False)
        aux_predict_output_noise = self.config["model"].get(
            "aux_predict_output_noise", False
        )
        num_inference_steps_training = training_cfg.get(
            "num_inference_steps_training", 1
        )

        desc = (
            f"Diffusion ({mode}) {epoch}/{total_epochs} Training"
            if training
            else f"Diffusion ({mode}) {epoch}/{total_epochs} Evaluating"
        )
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch.to(self.device)

            # 1. Inject Noise / Sample Parameters
            if mode == "diff-cont":
                # Sample alpha, sigma, scale for robust continuous diffusion
                # alpha controls the signal level (1.0 = clean truth, 0.0 = pure noise)
                # We use alpha_power and zero_signal_prob to bias training towards
                # hard "start from scratch" scenarios seen at inference start.
                alpha_power = training_cfg.get("alpha_power", 1.0)
                zero_signal_prob = training_cfg.get("zero_signal_prob", 0.0)
                sigma_max = training_cfg.get("sigma_max", 2.0)
                scale_min = training_cfg.get("scale_min", 4.0)
                scale_max = training_cfg.get("scale_max", 8.0)

                num_graphs = getattr(batch, "num_graphs", 1)

                # Sample Alpha (Signal)
                # We sample per-puzzle for maximum robustness
                alpha_rand = torch.rand(num_graphs, device=self.device)
                alphas = alpha_rand ** alpha_power

                # Zero-signal alignment (Tip 1): if alpha is 0, force sigma to sigma_max
                zero_mask = (
                    torch.rand(num_graphs, device=self.device) < zero_signal_prob
                )
                alphas[zero_mask] = 0.0

                # Sample Sigma (Noise)
                sigmas = torch.rand(num_graphs, device=self.device) * sigma_max
                sigmas[zero_mask] = sigma_max  # Align with start state

                # Sample Scale
                scale_range = scale_max - scale_min
                scales = (
                    torch.rand(num_graphs, device=self.device) * scale_range
                ) + scale_min

                data = inject_continuous_noise(
                    batch,
                    alpha=alphas,
                    sigma=sigmas,
                    scale=scales,
                    bridge_logits_idx=self.bridge_logits_idx,
                    model_config=self.config["model"],
                    device=self.device
                )
            elif mode == "flow-blind":
                num_graphs = getattr(batch, "num_graphs", 1)
                t_sampled = torch.rand((num_graphs, 1), device=self.device)

                data = inject_flow_noise(
                    batch,
                    t_sampled,
                    self.bridge_logits_idx,
                    self.config["model"],
                    training_cfg,
                    self.device
                )
                data.t_sampled = t_sampled
            else:
                # diff-discrete or fallback
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

            # --- Multi-Step Training Loop (f.6) ---
            step_losses = []

            # We clone the data to avoid modifying the original batch across steps
            # or in the loader.
            current_data = data
            current_input_noise = None
            if use_noise_head and mode == "diff-cont":
                # Initial noise: [sigma, alpha] from ground truth parameters
                current_input_noise = torch.stack([sigmas, alphas], dim=-1)

            for train_step in range(num_inference_steps_training):
                # 2. Forward Pass
                edge_attr = getattr(current_data, "edge_attr", None)
                model_has_verify = (
                    hasattr(self.model, "use_verification_head")
                    and self.model.use_verification_head
                )
                should_verify = use_verification and model_has_verify
                should_return_noise = (mode == "diff-cont") and use_noise_head

                # Input Noise Injection (f.7)
                time_input = None
                if mode == "flow-blind":
                    # Conditioning Augmentation: Add noise to time during training
                    time_noise_std = self.config["model"].get("time_noise_std", 0.1)
                    t_sampled = current_data.t_sampled
                    if training and time_noise_std > 0:
                        time_input = (
                            t_sampled + torch.randn_like(t_sampled) * time_noise_std
                        ).clamp(0, 1)
                    else:
                        time_input = t_sampled

                outputs = self.model(
                    current_data.x,
                    current_data.edge_index,
                    edge_attr=edge_attr,
                    batch=current_data.batch,
                    node_type=current_data.node_type,
                    return_verification=should_verify,
                    return_noise=should_return_noise,
                    input_noise=current_input_noise,
                    time=time_input,
                )

                # Unpack outputs based on what was requested
                verify_logits = None
                noise_pred = None
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    current_idx = 1
                    if should_verify:
                        verify_logits = outputs[current_idx]
                        current_idx += 1
                    if should_return_noise:
                        noise_pred = outputs[current_idx]
                else:
                    logits = outputs

                # 3. Loss Calculation
                edge_mask = current_data.edge_mask
                edge_batch = get_edge_batch_indices(current_data)
                node_type = getattr(current_data, "node_type", None)
                node_capacities = (
                    node_type if node_type is not None else current_data.x[:, 0].long()
                )
                edge_conflicts = getattr(current_data, "edge_conflicts", None)
                velocity_target = getattr(current_data, "velocity_target", None)

                # For flow-blind, auxiliary losses (degree, crossing) should be applied
                # to the predicted clean state, not the velocity.
                aux_logits = logits
                if mode == "flow-blind":
                    # Implied clean state c_hat = x_t + (1-t) * v
                    x_t = current_data.edge_attr[
                        :, self.bridge_logits_idx : self.bridge_logits_idx + 3
                    ]
                    t_edges = current_data.t_sampled[edge_batch]
                    aux_logits = x_t + (1.0 - t_edges) * logits

                losses = compute_combined_loss(
                    logits,
                    current_data.y,
                    current_data.edge_index,
                    node_capacities,
                    edge_conflicts,
                    edge_mask,
                    loss_weights,
                    verify_logits=verify_logits,
                    edge_batch=edge_batch,
                    velocity_target=velocity_target,
                    aux_logits=aux_logits,
                )
                loss = losses["total"]

                # Noise loss for diff-cont (Prophet Head)
                noise_loss_val = 0.0
                if mode == "diff-cont" and use_noise_head and noise_pred is not None:
                    # Target depends on aux_predict_output_noise
                    if aux_predict_output_noise:
                        # Target is estimated stats of the OUTPUT of the model
                        target_noise = estimate_signal_noise_stats(
                            logits, current_data.y, edge_batch, num_graphs, scale=scales
                        )
                    else:
                        # Target is the INPUT parameters
                        target_noise = torch.stack([sigmas, alphas], dim=-1)

                    noise_loss_val = torch.nn.functional.mse_loss(
                        noise_pred, target_noise
                    )
                    noise_weight = loss_weights.get("noise", 0.17)
                    loss += noise_weight * noise_loss_val

                step_losses.append(loss)

                if train_step < num_inference_steps_training - 1:
                    # Update board state for next step
                    with torch.no_grad():
                        # Update Input Noise Conditioning (Dynamic Alignment)
                        if (
                            mode == "diff-cont"
                            and use_noise_head
                            and noise_pred is not None
                        ):
                            current_input_noise = noise_pred.detach()

                        # Update board state
                        # We use the model's own predictions (Student Forcing)
                        # This preserves the heterogeneous noise structure (some edges
                        # confident, some not) which matches inference dynamics.

                        # Update edge logits
                        # Instead of re-sampling from GT, we take the model's current
                        # belief (logits), softmax and center it to stay in the
                        # trained manifold.
                        probs = torch.softmax(logits, dim=-1)
                        probs_centered = probs - (1.0 / 3.0)

                        target_state = probs_centered * scales[edge_batch].view(
                            -1, 1
                        )

                        # Move towards target (detached to avoid BPTT)
                        new_accumulated_logits = target_state.detach()

                        # Update current_data for next step
                        current_data = current_data.clone()
                        if self.bridge_logits_idx is not None:
                            current_data.edge_attr[
                                :, self.bridge_logits_idx:self.bridge_logits_idx + 3
                            ] = new_accumulated_logits

                        # Update node features
                        # Need to extract the current labels from the updated state
                        new_logits = current_data.edge_attr[
                            :, self.bridge_logits_idx:self.bridge_logits_idx + 3
                        ]
                        current_labels = new_logits.argmax(dim=-1).float()
                        current_data.x = update_node_features(
                            batch.x,  # Base capacities
                            current_labels,
                            current_data.edge_index,
                            current_data.node_type,
                            self.config["model"]
                        )

            total_batch_loss = torch.stack(step_losses).mean()

            if training and optimizer is not None:
                total_batch_loss.backward()
                optimizer.step()

            # 5. Metrics (from last step)
            total_loss += total_batch_loss.item()
            total_ce_loss += losses["ce"].item()
            total_degree_loss += losses["degree"].item()
            total_crossing_loss += losses["crossing"].item()
            total_verify_loss += losses["verify"].item()
            total_verify_acc += losses["verify_acc"].item()
            total_verify_recall_pos += losses["verify_recall_pos"].item()
            total_verify_recall_neg += losses["verify_recall_neg"].item()
            if losses["verify"] > 0:
                num_verify_batches += 1

            if isinstance(noise_loss_val, torch.Tensor):
                total_noise_loss += noise_loss_val.item()
            else:
                total_noise_loss += noise_loss_val

            total_steps += 1

            # Edge-wise accuracy
            puzzle_logits = aux_logits[edge_mask]
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
                    torch.ones(
                        len(puzzle_targets), dtype=torch.bool, device=self.device
                    ),
                    edge_batch[edge_mask],
                )
                total_solved_puzzles += num_perfect
                total_puzzles += num_total

        return {
            "loss": total_loss / total_steps if total_steps > 0 else 0.0,
            "ce_loss": total_ce_loss / total_steps if total_steps > 0 else 0.0,
            "degree_loss": total_degree_loss / total_steps if total_steps > 0 else 0.0,
            "crossing_loss": (
                total_crossing_loss / total_steps if total_steps > 0 else 0.0
            ),
            "verify_loss": total_verify_loss / total_steps if total_steps > 0 else 0.0,
            "verify_balanced_acc": (
                total_verify_acc / num_verify_batches
                if num_verify_batches > 0
                else 0.0
            ),
            "verify_recall_pos": (
                total_verify_recall_pos / num_verify_batches
                if num_verify_batches > 0
                else 0.0
            ),
            "verify_recall_neg": (
                total_verify_recall_neg / num_verify_batches
                if num_verify_batches > 0
                else 0.0
            ),
            "noise_loss": total_noise_loss / total_steps if total_steps > 0 else 0.0,
            "accuracy": (
                total_accuracy_accum / total_edges_count
                if total_edges_count > 0
                else 0.0
            ),
            "perfect_accuracy": (
                total_solved_puzzles / total_puzzles if total_puzzles > 0 else 0.0
            ),
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

        Returns
        -------
            dict: Metrics including accuracy and perfect puzzle accuracy at checkpoints.
        """
        if checkpoints is None:
            checkpoints = [1, 3, 5, 10, 20]

        self.model.eval()

        training_cfg = self.config["training"]
        model_cfg = self.config["model"]
        mode = training_cfg.get("mode", "diff-discrete").lower()
        diffusion_step_lr = training_cfg.get("diffusion_step_lr", 1.0)
        flush_first_step = training_cfg.get("flush_first_step", False)
        use_noise_head = model_cfg.get("use_noise_head", False)
        aux_predict_output_noise = model_cfg.get("aux_predict_output_noise", False)

        total_puzzles = 0
        puzzle_solved_at_k = dict.fromkeys(checkpoints, 0)
        final_accuracy_accum = 0.0
        total_batches = 0

        for batch in tqdm(loader, desc=f"Diffusion ({mode}) Rollout", leave=False):
            batch = batch.to(self.device)
            num_graphs = batch.num_graphs
            total_puzzles += num_graphs
            total_batches += 1

            # Start state initialization
            if mode == "diff-cont":
                # Initialize with noise to match the "alpha=0" training distribution.
                # We use sigma_max to start from the most noisy/neutral state
                # seen in training.
                sigma_max = training_cfg.get("sigma_max", 2.0)
                accumulated_logits = torch.randn(
                    (batch.edge_index.size(1), 3), device=self.device
                ) * sigma_max
            elif mode == "flow-blind":
                # Flow matching starts from noise at t=0
                # Use sigma_max from config, default to 2.0
                sigma_max = training_cfg.get("sigma_max", 2.0)
                accumulated_logits = torch.randn(
                    (batch.edge_index.size(1), 3), device=self.device
                ) * sigma_max
                current_t = torch.zeros((num_graphs, 1), device=self.device)
            else:
                # diff-discrete: Start with random bridges or empty?
                # Current logic starts with random if 100% noise
                current_bridges = torch.zeros(
                    batch.edge_index.size(1), device=self.device
                ).float()

                if hasattr(batch, "edge_mask") and batch.edge_mask is not None:
                    mask = batch.edge_mask
                    num_masked = mask.sum()
                    if num_masked > 0:
                        current_bridges[mask] = torch.randint(
                            0, 3, (num_masked,), device=self.device
                        ).float()

            # Track solved status per puzzle in batch
            edge_batch = get_edge_batch_indices(batch)
            puzzle_solved = torch.zeros(
                num_graphs, dtype=torch.bool, device=self.device
            )

            # Initialize input noise for Prophet Head feedback (f.7)
            # [sigma, alpha]. Start with max noise, zero signal.
            current_input_noise = None
            if use_noise_head:
                sigma_max = training_cfg.get("sigma_max", 2.0)
                current_input_noise = torch.zeros((num_graphs, 2), device=self.device)
                current_input_noise[:, 0] = sigma_max  # sigma
                current_input_noise[:, 1] = 0.0        # alpha

            # Working copy of data
            data = batch.clone()

            for step_idx in range(1, max_steps + 1):
                # 1. Update features based on current commitments
                if mode in ["diff-cont", "flow-blind"]:
                    current_labels = accumulated_logits.argmax(dim=-1).float()
                else:
                    current_labels = current_bridges

                if (
                    self.bridge_logits_idx is not None
                    and mode in ["diff-cont", "flow-blind"]
                ):
                    data.edge_attr[
                        :, self.bridge_logits_idx : self.bridge_logits_idx + 3
                    ] = accumulated_logits
                elif self.bridge_label_idx is not None:
                    data.edge_attr[:, self.bridge_label_idx] = current_labels
                    data.edge_attr[:, self.is_labeled_idx] = 1.0

                if self.config["model"].get("use_unused_capacity", True):
                    data.x = update_node_features(
                        batch.x,  # Original x with full capacities
                        current_labels,
                        data.edge_index,
                        data.node_type,
                        self.config["model"]
                    )

                with torch.no_grad():
                    should_return_noise = (
                        mode == "diff-cont"
                    ) and use_noise_head

                    time_val = None
                    if mode == "flow-blind":
                        time_val = current_t

                    outputs = self.model(
                        data.x,
                        data.edge_index,
                        edge_attr=data.edge_attr,
                        batch=data.batch,
                        node_type=data.node_type,
                        # rollout doesn't need verification
                        return_verification=False,
                        return_noise=should_return_noise,
                        input_noise=current_input_noise,
                        time=time_val,
                    )

                    pred_logits = None
                    noise_pred = None
                    if isinstance(outputs, tuple):
                        pred_logits = outputs[0]
                        # In rollout, return_verification is False,
                        # so if tuple, 2nd element must be noise if requested
                        if should_return_noise:
                            noise_pred = outputs[1]
                    else:
                        pred_logits = outputs

                    # 2. Update board state
                    if mode == "flow-blind":
                        # Blind Flow Update: board += Velocity * Step_Size
                        step_size = 1.0 / max_steps
                        # pred_logits is Velocity
                        accumulated_logits += pred_logits * step_size
                        # Update Clock
                        current_t = (current_t + step_size).clamp(0, 1)
                        current_labels = accumulated_logits.argmax(dim=-1).float()

                    elif mode == "diff-cont":
                        # Stable Attractor update: Project to signal space and
                        # move towards it. This prevents the accumulator from
                        # exploding into OOD values.

                        # Softmax + Center (matches the -1/3 shift in training)
                        probs = torch.softmax(pred_logits, dim=-1)
                        probs_centered = probs - (1.0 / 3.0)

                        # Scale to match the "Signal" magnitude from training
                        # We use scale_max to define the target confidence level
                        scale_max = training_cfg.get("scale_max", 8.0)
                        target_state = probs_centered * scale_max

                        # Exponential Moving Average (EMA) / "Diff" update
                        # moves the state lr percent of the way towards the target
                        use_adaptive_sampler = training_cfg.get(
                            "use_adaptive_sampler", False
                        )

                        effective_lr = diffusion_step_lr
                        if step_idx == 1 and flush_first_step:
                            effective_lr = 1.0
                        elif (
                            use_adaptive_sampler and current_input_noise is not None
                        ):
                            # Extract pred_alpha from current_input_noise[:, 1]
                            pred_alpha = current_input_noise[:, 1]
                            # Compute adaptive_lr = (1.0 - pred_alpha).clamp(...)
                            adaptive_lr = (1.0 - pred_alpha).clamp(
                                min=0.05, max=1.0
                            )
                            # Expand adaptive_lr to edge dimensions using edge_batch
                            # and add singleton dimension for broadcasting
                            effective_lr = adaptive_lr[edge_batch].view(-1, 1)

                        if (
                            isinstance(effective_lr, torch.Tensor)
                            or effective_lr >= 1.0
                        ):
                            if isinstance(effective_lr, torch.Tensor):
                                # Tensor-based update (adaptive)
                                accumulated_logits += effective_lr * (
                                    target_state - accumulated_logits
                                )
                            else:
                                # Scalar 1.0 update (flush)
                                accumulated_logits = target_state
                        else:
                            # Scalar LR update
                            accumulated_logits += effective_lr * (
                                target_state - accumulated_logits
                            )

                        current_labels = accumulated_logits.argmax(dim=-1).float()
                    else:
                        # Greedy update for discrete
                        current_bridges = pred_logits.argmax(dim=-1).float()
                        current_labels = current_bridges

                    # 3. Check which puzzles are solved
                    edge_mask = data.edge_mask
                    puzzle_targets = data.y[edge_mask]
                    puzzle_preds = current_labels[edge_mask]

                    # Check per puzzle
                    for i in range(num_graphs):
                        if puzzle_solved[i]:
                            continue

                        mask_i = (edge_batch[edge_mask] == i)
                        if torch.all(puzzle_preds[mask_i] == puzzle_targets[mask_i]):
                            puzzle_solved[i] = True

                    # 4. Update input noise for next step if using Prophet Head
                    # feedback. This is moved after the board update to ensure
                    # it reflects the CURRENT state.
                    if (
                        should_return_noise
                        and aux_predict_output_noise
                        and noise_pred is not None
                    ):
                        current_input_noise = noise_pred

                # Record checkpoints
                if step_idx in checkpoints:
                    puzzle_solved_at_k[step_idx] += puzzle_solved.sum().item()

                if puzzle_solved.all():
                    # If all solved, fill remaining checkpoints
                    for k in checkpoints:
                        if k > step_idx:
                            puzzle_solved_at_k[k] += puzzle_solved.sum().item()
                    break

            # Final accuracy for this batch
            edge_mask = data.edge_mask
            puzzle_targets = data.y[edge_mask]
            puzzle_preds = current_labels[edge_mask]
            final_accuracy_accum += (
                (puzzle_preds == puzzle_targets).float().mean().item()
            )

        # Aggregate results
        results = {
            f"perfect_acc_k{k}": (
                puzzle_solved_at_k[k] / total_puzzles if total_puzzles > 0 else 0.0
            )
            for k in checkpoints
        }
        results["accuracy"] = (
            final_accuracy_accum / total_batches if total_batches > 0 else 0.0
        )
        # Also include the last checkpoint as the general perfect_accuracy
        if checkpoints:
            results["perfect_accuracy"] = results[f"perfect_acc_k{checkpoints[-1]}"]

        return results
