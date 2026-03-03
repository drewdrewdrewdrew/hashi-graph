"""Diffusion trainer for Hashi Puzzle Solver."""

import torch
from typing import Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from .base import BaseTrainer, EpochMetrics
from ..utils.ar_utils import (
    get_edge_feature_indices,
    rewire_hierarchical_edges,
)
from ..utils.diffusion_utils import (
    estimate_signal_noise_stats,
    inject_continuous_noise,
    inject_flow_noise,
    inject_noise,
)
from ..losses.legacy import compute_combined_loss
from ..utils.train_utils import (
    calculate_batch_perfect_puzzles,
    get_edge_batch_indices,
    update_node_features,
)
from ..utils.common import custom_collate_with_conflicts


class DiffusionTrainer(BaseTrainer):
    """
    Trainer for Denoising Diffusion Hashi solving.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Edge feature indices
        edge_map = get_edge_feature_indices(self.config["model"])
        self.bridge_label_idx = edge_map.get("bridge_label")
        self.is_labeled_idx = edge_map.get("is_labeled")
        self.bridge_logits_idx = edge_map.get("bridge_logits")

        # Recursive batch carry-over buffers
        self.carry_over_buffer_train = []
        self.carry_over_buffer_val = []

    def _prepare_mixed_batch(
        self, batch: Any, training_cfg: dict[str, Any], training: bool = True
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Prepare a mixed batch of fresh and carried-over puzzles."""
        from typing import Any
        buffer = self.carry_over_buffer_train if training else self.carry_over_buffer_val
        batch_size = getattr(batch, "num_graphs", 1)
        zero_signal_prob = training_cfg.get("zero_signal_prob", 0.0)
        scale_min = training_cfg.get("scale_min", 4.0)
        scale_max = training_cfg.get("scale_max", 8.0)

        n_carry_target = int(batch_size * (1 - zero_signal_prob))
        n_carry = min(len(buffer), n_carry_target)
        n_fresh = batch_size - n_carry

        data_list = batch.to_data_list()
        fresh_puzzles = data_list[:n_fresh]

        sigma_max = training_cfg.get("sigma_max", 2.0)
        fresh_alphas = torch.zeros(n_fresh, device=self.device)
        fresh_sigmas = torch.full((n_fresh,), sigma_max, device=self.device)
        fresh_scales = (torch.rand(n_fresh, device=self.device) * (scale_max - scale_min)) + scale_min

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

            carry_stats_tensor = torch.stack(carry_stats_list)
            carry_sigmas_list = carry_stats_tensor[:, 0]
            carry_alphas_list = carry_stats_tensor[:, 1]
            carry_scales_list = torch.stack(carry_scales_list)

        combined_list = fresh_puzzles + carry_over_puzzles
        batch = custom_collate_with_conflicts(combined_list).to(self.device)

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
        from typing import Any
        buffer = self.carry_over_buffer_train if training else self.carry_over_buffer_val
        batch_size = getattr(batch, "num_graphs", 1)
        zero_signal_prob = training_cfg.get("zero_signal_prob", 0.0)
        n_carry_target = int(batch_size * (1 - zero_signal_prob))

        if n_carry_target <= 0:
            return

        probs = torch.softmax(logits, dim=-1)
        centered = probs - (1.0 / 3.0)
        edge_batch = get_edge_batch_indices(batch)
        next_input = centered * scales[edge_batch].view(-1, 1)

        next_input = next_input.detach()
        if noise_pred is not None:
            noise_pred = noise_pred.detach()

        data_list = batch.to_data_list()
        edge_counts = torch.zeros(batch_size, dtype=torch.long, device=self.device).scatter_add_(
            0, edge_batch, torch.ones_like(edge_batch, dtype=torch.long)
        )
        edge_ptr = torch.cat([torch.tensor([0], device=self.device), edge_counts.cumsum(0)])

        processed_puzzles = []
        for i, data in enumerate(data_list):
            start, end = edge_ptr[i], edge_ptr[i + 1]
            if self.bridge_logits_idx is not None:
                new_logits = next_input[start:end]
                data.edge_attr[:, self.bridge_logits_idx : self.bridge_logits_idx + 3] = new_logits
                if self.config["model"].get("use_unused_capacity", True):
                    current_labels = new_logits.argmax(dim=-1).float()
                    data.x = update_node_features(
                        data.x,
                        current_labels,
                        data.edge_index,
                        data.node_type,
                        self.config["model"]
                    )
            p_noise = noise_pred[i] if noise_pred is not None else torch.zeros((2,), device=self.device)
            p_scale = scales[i]
            processed_puzzles.append((data, p_noise, p_scale))

        if len(processed_puzzles) > n_carry_target:
            indices = torch.randperm(len(processed_puzzles))[:n_carry_target]
            sampled_puzzles = [processed_puzzles[i] for i in indices]
        else:
            sampled_puzzles = processed_puzzles

        buffer.extend(sampled_puzzles)
        max_buffer = batch_size * 4
        if len(buffer) > max_buffer:
            if training:
                self.carry_over_buffer_train = buffer[-max_buffer:]
            else:
                self.carry_over_buffer_val = buffer[-max_buffer:]

    def run_epoch(
        self,
        loader: DataLoader,
        training: bool = True,
        epoch: int = 1,
        total_epochs: int = 1,
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
        aux_predict_output_noise = self.config["model"].get("aux_predict_output_noise", False)
        num_inference_steps_training = training_cfg.get("num_inference_steps_training", 1)
        n_blocks = training_cfg.get("n_blocks")

        desc = f"Diffusion ({mode}) Epoch {epoch} " + ("Training" if training else "Evaluating")
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch.to(self.device)

            if mode == "diff-cont":
                use_carryover = training_cfg.get("recursive_carryover", False)
                if use_carryover:
                    batch, alphas, sigmas, scales, num_graphs = self._prepare_mixed_batch(batch, training_cfg, training=training)
                    data = batch
                else:
                    alpha_power = training_cfg.get("alpha_power", 1.0)
                    zero_signal_prob = training_cfg.get("zero_signal_prob", 0.0)
                    sigma_max = training_cfg.get("sigma_max", 2.0)
                    scale_min = training_cfg.get("scale_min", 4.0)
                    scale_max = training_cfg.get("scale_max", 8.0)
                    num_graphs = getattr(batch, "num_graphs", 1)
                    alphas = torch.rand(num_graphs, device=self.device) ** alpha_power
                    zero_mask = torch.rand(num_graphs, device=self.device) < zero_signal_prob
                    alphas[zero_mask] = 0.0
                    sigmas = torch.rand(num_graphs, device=self.device) * sigma_max
                    sigmas[zero_mask] = sigma_max
                    scales = (torch.rand(num_graphs, device=self.device) * (scale_max - scale_min)) + scale_min
                    data = inject_continuous_noise(batch, alpha=alphas, sigma=sigmas, scale=scales, bridge_logits_idx=self.bridge_logits_idx, model_config=self.config["model"], device=self.device)
            elif mode == "flow-blind":
                num_graphs = getattr(batch, "num_graphs", 1)
                t_sampled = torch.rand((num_graphs, 1), device=self.device)
                data = inject_flow_noise(batch, t_sampled, self.bridge_logits_idx, self.config["model"], training_cfg, self.device)
                data.t_sampled = t_sampled
            else:
                data = inject_noise(batch, noise_rate, self.bridge_label_idx, self.is_labeled_idx, self.config["model"], self.device)

            if training:
                self.optimizer.zero_grad()

            current_data = data
            current_input_noise = None
            if use_noise_head and mode == "diff-cont":
                current_input_noise = torch.stack([sigmas, alphas], dim=-1)

            step_losses = []
            step_ce_losses = []
            step_degree_losses = []
            step_crossing_losses = []
            step_verify_losses = []
            step_verify_accs = []
            step_verify_recall_pos = []
            step_verify_recall_neg = []
            step_noise_losses = []

            for train_step in range(num_inference_steps_training):
                if self.config["model"].get("use_component_meta", False):
                    curr_logits = None
                    if mode in ["diff-cont", "flow-blind"]:
                        curr_logits = current_data.edge_attr[:, self.bridge_logits_idx : self.bridge_logits_idx + 3]
                        rewire_bridges = curr_logits.argmax(dim=-1).float()
                    else:
                        rewire_bridges = current_data.edge_attr[:, self.bridge_label_idx]
                    current_data = rewire_hierarchical_edges(current_data, self.config["model"], current_bridges=rewire_bridges, logits=curr_logits)

                edge_attr = getattr(current_data, "edge_attr", None)
                model_has_verify = hasattr(self.model, "use_verification_head") and self.model.use_verification_head
                should_verify = use_verification and model_has_verify
                should_return_noise = (mode == "diff-cont") and use_noise_head

                time_input = None
                if mode == "flow-blind":
                    time_noise_std = self.config["model"].get("time_noise_std", 0.1)
                    t_sampled = current_data.t_sampled
                    time_input = (t_sampled + torch.randn_like(t_sampled) * time_noise_std).clamp(0, 1) if training and time_noise_std > 0 else t_sampled

                outputs = self.model(
                    current_data.x,
                    current_data.edge_index,
                    edge_attr=edge_attr,
                    edge_type=getattr(current_data, "edge_type", None),
                    batch=current_data.batch,
                    node_type=current_data.node_type,
                    return_verification=should_verify,
                    return_noise=should_return_noise,
                    input_noise=current_input_noise,
                    time=time_input,
                )

                verify_logits = None
                noise_pred = None
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    curr_idx = 1
                    if should_verify:
                        verify_logits = outputs[curr_idx]
                        curr_idx += 1
                    if should_return_noise:
                        noise_pred = outputs[curr_idx]
                else:
                    logits = outputs

                edge_mask = current_data.edge_mask
                edge_batch = get_edge_batch_indices(current_data)
                node_type = getattr(current_data, "node_type", None)
                node_capacities = node_type if node_type is not None else current_data.x[:, 0].long()
                edge_conflicts = getattr(current_data, "edge_conflict_index", None)
                velocity_target = getattr(current_data, "velocity_target", None)

                aux_logits = logits
                if mode == "flow-blind":
                    x_t = current_data.edge_attr[:, self.bridge_logits_idx : self.bridge_logits_idx + 3]
                    t_edges = current_data.t_sampled[edge_batch]
                    aux_logits = x_t + (1.0 - t_edges) * logits

                # Using legacy compute_combined_loss as per plan
                losses = compute_combined_loss(
                    logits, current_data.y, current_data.edge_index, node_capacities, edge_conflicts, edge_mask, loss_weights,
                    verify_logits=verify_logits, edge_batch=edge_batch, velocity_target=velocity_target, aux_logits=aux_logits
                )
                loss = losses["total"]

                noise_loss_val = 0.0
                if mode == "diff-cont" and use_noise_head and noise_pred is not None:
                    target_noise = estimate_signal_noise_stats(logits, current_data.y, edge_batch, num_graphs, scale=scales) if aux_predict_output_noise else torch.stack([sigmas, alphas], dim=-1)
                    noise_loss_val = torch.nn.functional.mse_loss(noise_pred, target_noise)
                    loss += loss_weights.get("noise", 0.17) * noise_loss_val

                step_losses.append(loss)
                step_ce_losses.append(losses["ce"])
                step_degree_losses.append(losses["degree"])
                step_crossing_losses.append(losses["crossing"])
                step_verify_losses.append(losses["verify"])
                step_verify_accs.append(losses["verify_acc"])
                step_verify_recall_pos.append(losses["verify_recall_pos"])
                step_verify_recall_neg.append(losses["verify_recall_neg"])
                step_noise_losses.append(noise_loss_val if isinstance(noise_loss_val, torch.Tensor) else torch.tensor(noise_loss_val, device=self.device))

                if train_step < num_inference_steps_training - 1:
                    with torch.no_grad():
                        if mode == "diff-cont" and use_noise_head and noise_pred is not None:
                            current_input_noise = noise_pred.detach()
                        probs = torch.softmax(logits, dim=-1)
                        probs_centered = probs - (1.0 / 3.0)
                        target_state = probs_centered * scales[edge_batch].view(-1, 1)
                        new_accumulated_logits = target_state.detach()
                        current_data = current_data.clone()
                        if self.bridge_logits_idx is not None:
                            current_data.edge_attr[:, self.bridge_logits_idx:self.bridge_logits_idx + 3] = new_accumulated_logits
                        new_logits = current_data.edge_attr[:, self.bridge_logits_idx:self.bridge_logits_idx + 3]
                        current_labels = new_logits.argmax(dim=-1).float()
                        current_data.x = update_node_features(batch.x, current_labels, current_data.edge_index, current_data.node_type, self.config["model"])

                        if training and n_blocks is not None and train_step == 0:
                            num_graphs_current = getattr(current_data, "num_graphs", 1)
                            if num_graphs_current > 1:
                                denom = (num_inference_steps_training - 1)
                                subsample_size = max(1, int(num_graphs_current * n_blocks / denom))
                                if subsample_size < num_graphs_current:
                                    indices = torch.randperm(num_graphs_current, device=self.device)[:subsample_size]
                                    batch = custom_collate_with_conflicts(batch[indices])
                                    current_data = custom_collate_with_conflicts(current_data[indices])
                                    if mode == "diff-cont":
                                        alphas, sigmas, scales = alphas[indices], sigmas[indices], scales[indices]
                                        num_graphs = subsample_size
                                    if current_input_noise is not None:
                                        current_input_noise = current_input_noise[indices]

            total_batch_loss = torch.stack(step_losses).mean()
            if training:
                total_batch_loss.backward()
                self.optimizer.step()

            use_carryover = training_cfg.get("recursive_carryover", False)
            if mode == "diff-cont" and use_carryover:
                self._refill_buffer(batch, logits, scales, training_cfg, noise_pred=noise_pred, training=training)

            total_loss += total_batch_loss.item()
            total_ce_loss += torch.stack(step_ce_losses).mean().item()
            total_degree_loss += torch.stack(step_degree_losses).mean().item()
            total_crossing_loss += torch.stack(step_crossing_losses).mean().item()
            total_verify_loss += torch.stack(step_verify_losses).mean().item()
            total_verify_acc += torch.stack(step_verify_accs).mean().item()
            total_verify_recall_pos += torch.stack(step_verify_recall_pos).mean().item()
            total_verify_recall_neg += torch.stack(step_verify_recall_neg).mean().item()
            if total_verify_loss > 0: num_verify_batches += 1
            total_noise_loss += torch.stack(step_noise_losses).mean().item()
            total_steps += 1

            edge_mask = current_data.edge_mask
            edge_batch = get_edge_batch_indices(current_data)
            puzzle_logits = aux_logits[edge_mask]
            puzzle_targets = current_data.y[edge_mask]
            with torch.no_grad():
                pred = puzzle_logits.argmax(dim=-1)
                total_accuracy_accum += (pred == puzzle_targets).float().mean().item()
                total_edges_count += 1
                _, num_perfect, num_total = calculate_batch_perfect_puzzles(puzzle_logits, puzzle_targets, torch.ones(len(puzzle_targets), dtype=torch.bool, device=self.device), edge_batch[edge_mask])
                total_solved_puzzles += num_perfect
                total_puzzles += num_total

        results = {
            "loss": total_loss / total_steps if total_steps > 0 else 0.0,
            "ce_loss": total_ce_loss / total_steps if total_steps > 0 else 0.0,
            "degree_loss": total_degree_loss / total_steps if total_steps > 0 else 0.0,
            "crossing_loss": total_crossing_loss / total_steps if total_steps > 0 else 0.0,
            "verify_loss": total_verify_loss / total_steps if total_steps > 0 else 0.0,
            "verify_balanced_acc": total_verify_acc / num_verify_batches if num_verify_batches > 0 else 0.0,
            "verify_recall_pos": total_verify_recall_pos / num_verify_batches if num_verify_batches > 0 else 0.0,
            "verify_recall_neg": total_verify_recall_neg / num_verify_batches if num_verify_batches > 0 else 0.0,
            "noise_loss": total_noise_loss / total_steps if total_steps > 0 else 0.0,
            "accuracy": total_accuracy_accum / total_edges_count if total_edges_count > 0 else 0.0,
            "perfect_accuracy": total_solved_puzzles / total_puzzles if total_puzzles > 0 else 0.0,
        }
        return results

    def run_rollout(self, loader: DataLoader, max_steps: int = 20, checkpoints: list[int] | None = None) -> dict[str, Any]:
        """Perform iterative cleanup (inference) on a batch of puzzles."""
        if checkpoints is None: checkpoints = [1, 3, 5, 10, 20]
        self.model.eval()
        training_cfg, model_cfg = self.config["training"], self.config["model"]
        mode = training_cfg.get("mode", "diff-discrete").lower()
        diffusion_step_lr, flush_first_step = training_cfg.get("diffusion_step_lr", 1.0), training_cfg.get("flush_first_step", False)
        use_noise_head, aux_predict_output_noise = model_cfg.get("use_noise_head", False), model_cfg.get("aux_predict_output_noise", False)

        total_puzzles, puzzle_solved_at_k = 0, dict.fromkeys(checkpoints, 0)
        final_accuracy_accum, total_batches = 0.0, 0

        for batch in tqdm(loader, desc=f"Diffusion ({mode}) Rollout", leave=False):
            batch = batch.to(self.device)
            num_graphs = batch.num_graphs
            total_puzzles += num_graphs
            total_batches += 1

            if mode in ["diff-cont", "flow-blind"]:
                sigma_max = training_cfg.get("sigma_max", 2.0)
                accumulated_logits = torch.randn((batch.edge_index.size(1), 3), device=self.device) * sigma_max
                if mode == "flow-blind": current_t = torch.zeros((num_graphs, 1), device=self.device)
            else:
                current_bridges = torch.zeros(batch.edge_index.size(1), device=self.device).float()
                if hasattr(batch, "edge_mask") and batch.edge_mask is not None:
                    mask = batch.edge_mask
                    if mask.sum() > 0: current_bridges[mask] = torch.randint(0, 3, (mask.sum(),), device=self.device).float()

            edge_batch = get_edge_batch_indices(batch)
            puzzle_solved = torch.zeros(num_graphs, dtype=torch.bool, device=self.device)
            current_input_noise = None
            if use_noise_head:
                current_input_noise = torch.zeros((num_graphs, 2), device=self.device)
                current_input_noise[:, 0] = training_cfg.get("sigma_max", 2.0)

            for step_idx in range(1, max_steps + 1):
                data = batch.clone()
                current_labels = accumulated_logits.argmax(dim=-1).float() if mode in ["diff-cont", "flow-blind"] else current_bridges
                if self.bridge_logits_idx is not None and mode in ["diff-cont", "flow-blind"]:
                    data.edge_attr[:, self.bridge_logits_idx : self.bridge_logits_idx + 3] = accumulated_logits
                elif self.bridge_label_idx is not None:
                    data.edge_attr[:, self.bridge_label_idx] = current_labels
                    data.edge_attr[:, self.is_labeled_idx] = 1.0
                if self.config["model"].get("use_unused_capacity", True):
                    data.x = update_node_features(batch.x, current_labels, data.edge_index, data.node_type, self.config["model"])
                if self.config["model"].get("use_component_meta", False):
                    data = rewire_hierarchical_edges(data, self.config["model"], current_bridges=current_labels, logits=accumulated_logits if mode in ["diff-cont", "flow-blind"] else None)

                with torch.no_grad():
                    should_return_noise = (mode == "diff-cont") and use_noise_head
                    outputs = self.model(data.x, data.edge_index, edge_attr=data.edge_attr, edge_type=getattr(data, "edge_type", None), batch=data.batch, node_type=data.node_type, return_verification=False, return_noise=should_return_noise, input_noise=current_input_noise, time=current_t if mode == "flow-blind" else None)
                    current_edge_batch, num_orig_edges = get_edge_batch_indices(data), accumulated_logits.size(0)
                    pred_logits, noise_pred = (outputs[0], outputs[1]) if isinstance(outputs, tuple) else (outputs, None)

                    if mode == "flow-blind":
                        step_size = 1.0 / max_steps
                        accumulated_logits += pred_logits[:num_orig_edges] * step_size
                        current_t = (current_t + step_size).clamp(0, 1)
                        current_labels = accumulated_logits.argmax(dim=-1).float()
                    elif mode == "diff-cont":
                        probs_centered = torch.softmax(pred_logits, dim=-1) - (1.0 / 3.0)
                        target_state = probs_centered * training_cfg.get("scale_max", 8.0)
                        effective_lr = 1.0 if step_idx == 1 and flush_first_step else ( (1.0 - current_input_noise[:, 1]).clamp(min=0.05, max=1.0)[current_edge_batch].view(-1, 1) if training_cfg.get("use_adaptive_sampler", False) and current_input_noise is not None else diffusion_step_lr )
                        accumulated_logits += (effective_lr[:num_orig_edges] if isinstance(effective_lr, torch.Tensor) else effective_lr) * (target_state[:num_orig_edges] - accumulated_logits)
                        current_labels = accumulated_logits.argmax(dim=-1).float()
                    else:
                        current_bridges = pred_logits[:num_orig_edges].argmax(dim=-1).float()
                        current_labels = current_bridges

                    edge_mask = data.edge_mask
                    puzzle_targets, puzzle_preds = data.y[edge_mask], current_labels[edge_mask[:num_orig_edges]]
                    for i in range(num_graphs):
                        if not puzzle_solved[i]:
                            mask_i = (current_edge_batch[edge_mask] == i)
                            if torch.all(puzzle_preds[mask_i] == puzzle_targets[mask_i]): puzzle_solved[i] = True
                    if should_return_noise and aux_predict_output_noise and noise_pred is not None: current_input_noise = noise_pred

                if step_idx in checkpoints: puzzle_solved_at_k[step_idx] += puzzle_solved.sum().item()
                if puzzle_solved.all():
                    for k in checkpoints:
                        if k > step_idx: puzzle_solved_at_k[k] += puzzle_solved.sum().item()
                    break
            
            edge_mask = data.edge_mask
            final_accuracy_accum += (current_labels[edge_mask[:accumulated_logits.size(0)]] == data.y[edge_mask]).float().mean().item()

        results = {f"perfect_acc_k{k}": (puzzle_solved_at_k[k] / total_puzzles if total_puzzles > 0 else 0.0) for k in checkpoints}
        results["accuracy"] = final_accuracy_accum / total_batches if total_batches > 0 else 0.0
        if checkpoints: results["perfect_accuracy"] = results[f"perfect_acc_k{checkpoints[-1]}"]
        return results
