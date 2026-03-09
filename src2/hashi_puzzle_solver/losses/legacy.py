"""Legacy loss functions for Hashi puzzle constraint learning."""

import torch
import torch.nn.functional as functional
from torch_geometric.utils import scatter


def compute_degree_violation_loss(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    node_capacities: torch.Tensor,
    edge_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Legacy degree violation loss."""
    probs = functional.softmax(logits, dim=-1)
    bridge_values = (
        probs[:, 0] * 0.0 + probs[:, 1] * 1.0 + probs[:, 2] * 2.0
    )
    bridge_values_masked = bridge_values * edge_mask.float()
    src_nodes = edge_index[0]
    predicted_degrees = scatter(
        bridge_values_masked, src_nodes, dim=0, reduce="sum",
    )
    target_degrees = node_capacities.float()
    is_puzzle_node = (node_capacities >= 1) & (node_capacities <= 8)
    target_degrees = target_degrees * is_puzzle_node.float()
    if predicted_degrees.size(0) < target_degrees.size(0):
        padding = torch.zeros(
            target_degrees.size(0) - predicted_degrees.size(0),
            device=predicted_degrees.device,
            dtype=predicted_degrees.dtype,
        )
        predicted_degrees = torch.cat([predicted_degrees, padding])
    return functional.mse_loss(predicted_degrees, target_degrees, reduction=reduction)


def compute_crossing_loss(
    logits: torch.Tensor,
    edge_conflict_index: torch.Tensor | None,
    _edge_mask: torch.Tensor,
    reduction: str = "mean",
    mode: str = "multiplicative",
) -> torch.Tensor:
    """Legacy crossing loss."""
    if edge_conflict_index is None or edge_conflict_index.size(1) == 0:
        return torch.tensor(0.0, device=logits.device)
    e1_indices = edge_conflict_index[0]
    e2_indices = edge_conflict_index[1]
    probs = functional.softmax(logits, dim=-1)
    bridge_exists_prob = probs[:, 1] + probs[:, 2]
    prob1 = bridge_exists_prob[e1_indices]
    prob2 = bridge_exists_prob[e2_indices]
    if mode == "multiplicative":
        crossing_losses = prob1 * prob2
    elif mode == "max_product":
        max_logit1 = logits[e1_indices].max(dim=-1).values
        max_logit2 = logits[e2_indices].max(dim=-1).values
        crossing_losses = functional.relu(max_logit1) * functional.relu(max_logit2)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if reduction == "mean":
        return crossing_losses.mean()
    if reduction == "sum":
        return crossing_losses.sum()
    raise ValueError(f"Unknown reduction: {reduction}")


def compute_verification_loss(
    verify_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    targets: torch.Tensor,
    edge_mask: torch.Tensor,
    edge_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy verification loss."""
    if verify_logits is None or verify_logits.numel() == 0:
        device = edge_logits.device
        return (
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        )
    edge_preds = edge_logits[edge_mask].argmax(dim=-1)
    targets_original = targets[edge_mask]
    edge_batch_original = edge_batch[edge_mask]
    edge_incorrect = (edge_preds != targets_original).long()
    num_puzzles = edge_batch_original.max().item() + 1
    errors_per_puzzle = scatter(
        edge_incorrect, edge_batch_original, dim=0, dim_size=num_puzzles, reduce="sum",
    )
    verify_targets = (errors_per_puzzle == 0).float().unsqueeze(-1)
    num_pos = verify_targets.sum()
    pos_weight = (float(num_puzzles) - num_pos + 1.0) / (num_pos + 1.0)
    loss = functional.binary_cross_entropy_with_logits(
        verify_logits, verify_targets, pos_weight=pos_weight,
    )
    verify_preds = (torch.sigmoid(verify_logits) > 0.5).float()
    pos_mask = verify_targets == 1.0
    neg_mask = verify_targets == 0.0
    num_pos = pos_mask.sum()
    num_neg = neg_mask.sum()
    recall_pos = (
        (verify_preds[pos_mask] == 1.0).float().mean()
        if num_pos > 0
        else torch.tensor(1.0, device=verify_logits.device)
    )
    recall_neg = (
        (verify_preds[neg_mask] == 0.0).float().mean()
        if num_neg > 0
        else torch.tensor(1.0, device=verify_logits.device)
    )
    balanced_acc = (recall_pos + recall_neg) / 2.0
    return loss, balanced_acc, recall_pos, recall_neg


def compute_combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    edge_index: torch.Tensor,
    node_capacities: torch.Tensor,
    edge_conflict_index: torch.Tensor | None,
    edge_mask: torch.Tensor,
    loss_weights: dict[str, float] | None = None,
    verify_logits: torch.Tensor | None = None,
    edge_batch: torch.Tensor | None = None,
    velocity_target: torch.Tensor | None = None,
    aux_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Legacy combined loss."""
    if loss_weights is None:
        loss_weights = {"ce": 1.0, "degree": 0.1, "crossing": 0.5, "verify": 0.0}
    if aux_logits is None:
        aux_logits = logits
    if velocity_target is not None:
        loss_ce = functional.mse_loss(logits, velocity_target, reduction="mean")
    else:
        logits_original = logits[edge_mask]
        targets_original = targets[edge_mask]
        loss_ce = functional.cross_entropy(logits_original, targets_original, reduction="mean")
    loss_degree = compute_degree_violation_loss(aux_logits, edge_index, node_capacities, edge_mask)
    loss_crossing = compute_crossing_loss(aux_logits, edge_conflict_index, edge_mask)
    verify_weight = loss_weights.get("verify", 0.0)
    if verify_logits is not None and edge_batch is not None and verify_weight > 0:
        loss_verify, verify_acc, verify_recall_pos, verify_recall_neg = compute_verification_loss(
            verify_logits, aux_logits, targets, edge_mask, edge_batch
        )
    else:
        loss_verify = torch.tensor(0.0, device=logits.device)
        verify_acc = torch.tensor(0.0, device=logits.device)
        verify_recall_pos = torch.tensor(0.0, device=logits.device)
        verify_recall_neg = torch.tensor(0.0, device=logits.device)
    total_loss = (
        loss_weights["ce"] * loss_ce
        + loss_weights.get("degree", 0.0) * loss_degree
        + loss_weights.get("crossing", 0.0) * loss_crossing
        + verify_weight * loss_verify
    )
    return {
        "total": total_loss, "ce": loss_ce, "degree": loss_degree,
        "crossing": loss_crossing, "verify": loss_verify,
        "verify_acc": verify_acc, "verify_recall_pos": verify_recall_pos,
        "verify_recall_neg": verify_recall_neg,
    }
