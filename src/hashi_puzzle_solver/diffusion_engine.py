"""Diffusion training engine for Hashi GNN."""

from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ar_utils import (
    get_edge_feature_indices,
    rewire_hierarchical_edges,
)
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
        self.bridge_logits_idx = edge_map.get("bridge_logits")

        # Recursive batch carry-over buffers (Item L)
        # Separate buffers for train and validation to avoid data leakage.
        self.carry_over_buffer_train = []  # List[Data]
        self.carry_over_buffer_val = []    # List[Data]

    def _prepare_mixed_batch(
        self, batch: Any, training_cfg: dict[str, Any], training: bool = True
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Prepare a mixed batch of fresh and carried-over puzzles."""
        buffer = (
            self.carry_over_buffer_train if training 
            else self.carry_over_buffer_val
        )
        batch_size = getattr(batch, "num_graphs", 1)
        zero_signal_prob = training_cfg.get("zero_signal_prob", 0.0)
        sigma_max = training_cfg.get("sigma_max", 2.0)
        scale_min = training_cfg.get("scale_min", 4.0)
        scale_max = training_cfg.get("scale_max", 8.0)
        alpha_power = training_cfg.get("alpha_power", 1.0)

        # 1. Calculate Split
        n_carry_target = int(batch_size * (1 - zero_signal_prob))
        n_carry = min(len(buffer), n_carry_target)
        n_fresh = batch_size - n_carry

        # 2. Prepare Fresh Data
        data_list = batch.to_data_list()
        fresh_puzzles = data_list[:n_fresh]

        # Sample fresh params
        fresh_alphas = torch.rand(n_fresh, device=self.device) ** alpha_power
        # Force fresh puzzles to start noise (alpha=0, sigma=sigma_max)
        fresh_alphas[:] = 0.0
        fresh_sigmas = torch.full((n_fresh,), sigma_max, device=self.device)
        fresh_scales = (
            torch.rand(n_fresh, device=self.device) * (scale_max - scale_min)
        ) + scale_min

        # Inject noise into fresh puzzles
        # We need to temporarily batch them for inject_continuous_noise
        if n_fresh > 0:
            fresh_batch = custom_collate_with_conflicts(fresh_puzzles).to(self.device)
            fresh_batch = inject_continuous_noise(
                fresh_batch,
                alpha=fresh_alphas,
                sigma=fresh_sigmas,
                scale=fresh_scales,
                bridge_logits_idx=self.bridge_logits_idx,
                model_config=self.config["model"],
                device=self.device,
            )
            fresh_puzzles = fresh_batch.to_data_list()

        # 3. Prepare Carry-Over Data
        carry_over_puzzles = []
        carry_alphas_list = []
        carry_sigmas_list = []
        carry_scales_list = []

        if n_carry > 0:
            carry_stats_list = []
            for _ in range(n_carry):
                data, noise_stats, scale = buffer.pop(0)
                carry_over_puzzles.append(data)
                carry_stats_list.append(noise_stats)
                carry_scales_list.append(scale)

            # USE PREDICTED NOISE (Student Forcing / Deep Rollout)
            # This is the fix: we use the model's own assessment from the previous step.
            carry_stats_tensor = torch.stack(carry_stats_list)
            carry_sigmas_list = carry_stats_tensor[:, 0]
            carry_alphas_list = carry_stats_tensor[:, 1]
            carry_scales_list = torch.stack(carry_scales_list)

        # 4. Merge and Collate
        combined_list = fresh_puzzles + carry_over_puzzles
        batch = custom_collate_with_conflicts(combined_list).to(self.device)

        # Concatenate params
        alphas = fresh_alphas
        sigmas = fresh_sigmas
        scales = fresh_scales

        if n_carry > 0:
            alphas = torch.cat([alphas, carry_alphas_list])
            sigmas = torch.cat([sigmas, carry_sigmas_list])
            scales = torch.cat([scales, carry_scales_list])

        return batch, alphas, sigmas, scales, len(combined_list)

    def _refill_buffer(
        self,
        batch: Any,
        logits: torch.Tensor,
        scales: torch.Tensor,
        training_cfg: dict[str, Any],
        noise_pred: torch.Tensor | None = None,
        training: bool = True,
    ) -> None:
        """Process output logits and refill the carry-over buffer."""
        buffer = (
            self.carry_over_buffer_train if training 
            else self.carry_over_buffer_val
        )
        batch_size = getattr(batch, "num_graphs", 1)
        zero_signal_prob = training_cfg.get("zero_signal_prob", 0.0)
        n_carry_target = int(batch_size * (1 - zero_signal_prob))

        if n_carry_target <= 0:
            return

        # 1. Process Output Logits
        # Convert to next input format: softmax -> center -> scale
        probs = torch.softmax(logits, dim=-1)
        centered = probs - (1.0 / 3.0)

        edge_batch = get_edge_batch_indices(batch)
        next_input = centered * scales[edge_batch].view(-1, 1)

        # 2. Update Data Objects
        # We need to detach next_input to avoid memory leaks
        next_input = next_input.detach()
        if noise_pred is not None:
            noise_pred = noise_pred.detach()

        data_list = batch.to_data_list()

        # Calculate edge counts per graph
        edge_counts = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        ).scatter_add_(
            0, edge_batch, torch.ones_like(edge_batch, dtype=torch.long)
        )
        edge_ptr = torch.cat(
            [torch.tensor([0], device=self.device), edge_counts.cumsum(0)]
        )

        # Prepare tuples to store in buffer: (Data, noise_pred_i, scale_i)
        processed_puzzles = []

        for i, data in enumerate(data_list):
            start, end = edge_ptr[i], edge_ptr[i + 1]
            if self.bridge_logits_idx is not None:
                new_logits = next_input[start:end]
                data.edge_attr[
                    :, self.bridge_logits_idx : self.bridge_logits_idx + 3
                ] = new_logits

                # NEW: Update node features to be consistent with the new logits
                if self.config["model"].get("use_unused_capacity", True):
                    current_labels = new_logits.argmax(dim=-1).float()
                    data.x = update_node_features(
                        data.x,  # Note: this data.x already has base capacities
                        current_labels,
                        data.edge_index,
                        data.node_type,
                        self.config["model"]
                    )

            # Capture noise prediction and scale for this puzzle
            # If noise_pred is None, we use a zero tensor as a neutral fallback
            if noise_pred is not None:
                p_noise = noise_pred[i]
            else:
                p_noise = torch.zeros((2,), device=self.device)
            
            p_scale = scales[i]
            processed_puzzles.append((data, p_noise, p_scale))

        # 3. Random Sampling
        if len(processed_puzzles) > n_carry_target:
            indices = torch.randperm(len(processed_puzzles))[:n_carry_target]
            sampled_puzzles = [processed_puzzles[i] for i in indices]
        else:
            sampled_puzzles = processed_puzzles

        buffer.extend(sampled_puzzles)

        # Optional: Limit buffer size to avoid excessive memory usage
        # Let's keep it at 2x batch_size or similar if needed,
        # but for now we just follow the plan of sampling n_carry_target.
        max_buffer = batch_size * 4
        if len(buffer) > max_buffer:
            # Re-assigning to avoid issues with slice assignments if needed,
            # but list slice works fine.
            if training:
                self.carry_over_buffer_train = buffer[-max_buffer:]
            else:
                self.carry_over_buffer_val = buffer[-max_buffer:]


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
        n_blocks = training_cfg.get("n_blocks")

        desc = (
            f"Diffusion ({mode}) {epoch}/{total_epochs} Training"
            if training
            else f"Diffusion ({mode}) {epoch}/{total_epochs} Evaluating"
        )
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch.to(self.device)

            # 1. Inject Noise / Sample Parameters
            if mode == "diff-cont":
                use_carryover = training_cfg.get("recursive_carryover", False)
                if use_carryover:
                    # Recursive Batch Carry-Over (Item L)
                    # Used in both training and epoch validation for consistency.
                    (
                        batch,
                        alphas,
                        sigmas,
                        scales,
                        num_graphs,
                    ) = self._prepare_mixed_batch(
                        batch, training_cfg, training=training
                    )
                    data = batch
                else:
                    # Standard evaluation noise logic or if carryover is disabled
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
                        device=self.device,
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
            step_ce_losses = []
            step_degree_losses = []
            step_crossing_losses = []
            step_verify_losses = []
            step_verify_accs = []
            step_verify_recall_pos = []
            step_verify_recall_neg = []
            step_noise_losses = []

            # We clone the data to avoid modifying the original batch across steps
            # or in the loader.
            current_data = data
            current_input_noise = None
            if use_noise_head and mode == "diff-cont":
                # Initial noise: [sigma, alpha] from ground truth parameters
                current_input_noise = torch.stack([sigmas, alphas], dim=-1)

            for train_step in range(num_inference_steps_training):
                # --- Hierarchical Rewiring ---
                if self.config["model"].get("use_component_meta", False):
                    # Get current bridges for topology detection
                    curr_logits = None
                    if mode in ["diff-cont", "flow-blind"]:
                        # Extract from continuous logits
                        curr_logits = current_data.edge_attr[
                            :, self.bridge_logits_idx : self.bridge_logits_idx + 3
                        ]
                        rewire_bridges = curr_logits.argmax(dim=-1).float()
                    else:
                        # Extract from discrete labels
                        rewire_bridges = current_data.edge_attr[
                            :, self.bridge_label_idx
                        ]

                    current_data = rewire_hierarchical_edges(
                        current_data,
                        self.config["model"],
                        current_bridges=rewire_bridges,
                        logits=curr_logits,
                    )

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
                edge_conflicts = getattr(current_data, "edge_conflict_index", None)
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
                            logits,
                            current_data.y,
                            edge_batch,
                            num_graphs,
                            scale=scales,
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
                step_ce_losses.append(losses["ce"])
                step_degree_losses.append(losses["degree"])
                step_crossing_losses.append(losses["crossing"])
                step_verify_losses.append(losses["verify"])
                step_verify_accs.append(losses["verify_acc"])
                step_verify_recall_pos.append(losses["verify_recall_pos"])
                step_verify_recall_neg.append(losses["verify_recall_neg"])
                step_noise_losses.append(
                    noise_loss_val if isinstance(noise_loss_val, torch.Tensor)
                    else torch.tensor(noise_loss_val, device=self.device)
                )

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

                        # Perform subsampling after first step if configured
                        # (Computational Budget Allocation)
                        if training and n_blocks is not None and train_step == 0:
                            num_graphs_current = getattr(current_data, "num_graphs", 1)
                            if num_graphs_current > 1:
                                # Subsample budget math: B' = B * n_blocks / (S - 1)
                                denom = (num_inference_steps_training - 1)
                                subsample_size = max(
                                    1, int(num_graphs_current * n_blocks / denom)
                                )

                                if subsample_size < num_graphs_current:
                                    indices = torch.randperm(
                                        num_graphs_current, device=self.device
                                    )[:subsample_size]

                                    # Slice batch and data objects
                                    # We slice both because batch.x is used for original
                                    # capacities. custom_collate_with_conflicts ensures
                                    # we get a Batch object back instead of a list.
                                    batch = custom_collate_with_conflicts(
                                        batch[indices]
                                    )
                                    current_data = custom_collate_with_conflicts(
                                        current_data[indices]
                                    )

                                    # Slice associated metadata
                                    if mode == "diff-cont":
                                        alphas = alphas[indices]
                                        sigmas = sigmas[indices]
                                        scales = scales[indices]
                                        # Update num_graphs for
                                        # estimate_signal_noise_stats
                                        # in next steps
                                        num_graphs = subsample_size
                                    elif mode == "flow-blind":
                                        # t_sampled is already stored in
                                        # current_data.t_sampled and index_select
                                        # handles it.
                                        pass

                                    if current_input_noise is not None:
                                        current_input_noise = current_input_noise[
                                            indices
                                        ]

            total_batch_loss = torch.stack(step_losses).mean()
            avg_ce_loss = torch.stack(step_ce_losses).mean()
            avg_degree_loss = torch.stack(step_degree_losses).mean()
            avg_crossing_loss = torch.stack(step_crossing_losses).mean()
            avg_verify_loss = torch.stack(step_verify_losses).mean()
            avg_verify_acc = torch.stack(step_verify_accs).mean()
            avg_verify_recall_pos = torch.stack(step_verify_recall_pos).mean()
            avg_verify_recall_neg = torch.stack(step_verify_recall_neg).mean()
            avg_noise_loss = torch.stack(step_noise_losses).mean()

            if training and optimizer is not None:
                total_batch_loss.backward()
                optimizer.step()

            # Refill carry-over buffer (Item L)
            # We refill in both training and epoch validation to maintain consistency.
            use_carryover = training_cfg.get("recursive_carryover", False)
            if mode == "diff-cont" and use_carryover:
                self._refill_buffer(
                    batch,
                    logits,
                    scales,
                    training_cfg,
                    noise_pred=noise_pred,
                    training=training,
                )

            # 5. Metrics (from last step)
            total_loss += total_batch_loss.item()
            total_ce_loss += avg_ce_loss.item()
            total_degree_loss += avg_degree_loss.item()
            total_crossing_loss += avg_crossing_loss.item()
            total_verify_loss += avg_verify_loss.item()
            total_verify_acc += avg_verify_acc.item()
            total_verify_recall_pos += avg_verify_recall_pos.item()
            total_verify_recall_neg += avg_verify_recall_neg.item()
            if avg_verify_loss > 0:
                num_verify_batches += 1

            total_noise_loss += avg_noise_loss.item()
            total_steps += 1

            # Edge-wise accuracy (Final Step Only)
            # We use current_data here because it might have been subsampled
            edge_mask = current_data.edge_mask
            edge_batch = get_edge_batch_indices(current_data)
            puzzle_logits = aux_logits[edge_mask]
            puzzle_targets = current_data.y[edge_mask]
            with torch.no_grad():
                pred = puzzle_logits.argmax(dim=-1)
                acc = (pred == puzzle_targets).float().mean().item()
                total_accuracy_accum += acc
                total_edges_count += 1

                # Perfect puzzle accuracy (Final Step Only)
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

            for step_idx in range(1, max_steps + 1):
                # Working copy of data - we start fresh from the original batch
                # each step to avoid accumulating hierarchical meta-edges.
                data = batch.clone()

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

                # --- Hierarchical Rewiring ---
                if self.config["model"].get("use_component_meta", False):
                    rewire_logits = None
                    if mode in ["diff-cont", "flow-blind"]:
                        rewire_logits = accumulated_logits

                    data = rewire_hierarchical_edges(
                        data,
                        self.config["model"],
                        current_bridges=current_labels,
                        logits=rewire_logits,
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

                    # 2. Update board state
                    # Recalculate edge_batch to match the current (possibly rewired) graph
                    current_edge_batch = get_edge_batch_indices(data)
                    num_orig_edges = accumulated_logits.size(0)

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

                    if mode == "flow-blind":
                        # Blind Flow Update: board += Velocity * Step_Size
                        step_size = 1.0 / max_steps
                        # pred_logits is Velocity
                        # Slice to original puzzle edges
                        accumulated_logits += pred_logits[:num_orig_edges] * step_size
                        # Update Clock
                        current_t = (current_t + step_size).clamp(0, 1)
                        current_labels = accumulated_logits.argmax(dim=-1).float()

                    elif mode == "diff-cont":
                        # Stable Attractor update
                        probs = torch.softmax(pred_logits, dim=-1)
                        probs_centered = probs - (1.0 / 3.0)

                        scale_max = training_cfg.get("scale_max", 8.0)
                        target_state = probs_centered * scale_max

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
                            adaptive_lr = (1.0 - pred_alpha).clamp(
                                min=0.05, max=1.0
                            )
                            # Use current_edge_batch to expand to all current edges
                            effective_lr = adaptive_lr[current_edge_batch].view(-1, 1)

                        if (
                            isinstance(effective_lr, torch.Tensor)
                            or effective_lr >= 1.0
                        ):
                            if isinstance(effective_lr, torch.Tensor):
                                # Tensor-based update (adaptive)
                                # Slice both target and lr to match accumulator
                                accumulated_logits += effective_lr[:num_orig_edges] * (
                                    target_state[:num_orig_edges] - accumulated_logits
                                )
                            else:
                                # Scalar 1.0 update (flush)
                                accumulated_logits = target_state[:num_orig_edges]
                        else:
                            # Scalar LR update
                            accumulated_logits += effective_lr * (
                                target_state[:num_orig_edges] - accumulated_logits
                            )

                        current_labels = accumulated_logits.argmax(dim=-1).float()
                    else:
                        # Greedy update for discrete
                        # Slice pred_logits to original edges
                        current_bridges = pred_logits[:num_orig_edges].argmax(dim=-1).float()
                        current_labels = current_bridges

                    # 3. Check which puzzles are solved
                    edge_mask = data.edge_mask
                    puzzle_targets = data.y[edge_mask]
                    # current_labels is num_orig_edges, edge_mask is num_current_edges
                    # But edge_mask only has True values in the first num_orig_edges.
                    puzzle_preds = current_labels[edge_mask[:num_orig_edges]]

                    # Check per puzzle
                    for i in range(num_graphs):
                        if puzzle_solved[i]:
                            continue

                        # Use current_edge_batch for the solve check
                        mask_i = (current_edge_batch[edge_mask] == i)
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
            puzzle_preds = current_labels[edge_mask[:accumulated_logits.size(0)]]
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
