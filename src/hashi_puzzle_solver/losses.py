"""
Auxiliary loss functions for Hashi puzzle constraint learning.

These losses encode domain-specific rules to guide the model toward valid solutions.
"""
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter
from typing import Dict, List, Optional, Tuple


def compute_degree_violation_loss(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    node_capacities: torch.Tensor,
    edge_mask: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Degree Violation Loss: Enforces that sum of bridges equals island capacity.
    
    For each island node, the sum of predicted bridge values on incident edges
    should equal the node's capacity (n). This is a fundamental Hashi constraint.
    
    Args:
        logits: Model output logits [num_edges, num_classes] for ALL edges (original + meta)
        edge_index: Graph connectivity [2, num_edges]
        node_capacities: Island capacity values [num_nodes] (the 'n' feature)
        edge_mask: Boolean mask [num_edges] indicating original puzzle edges (excludes meta/conflict)
        reduction: 'mean' or 'sum' for loss aggregation
    
    Returns:
        torch.Tensor: Scalar loss value
        
    Implementation:
        For each node, compute predicted degree as:
            predicted_degree = sum(predicted_bridge_values for incident original edges)
        
        Loss = MSE(predicted_degree, target_capacity)
        
        Why MSE: Penalizes large violations more than small ones, encouraging
        the model to get close to the target even if not exact initially.
        
    Note: Only original puzzle edges (edge_mask=True) contribute to degree counting.
          Meta and conflict edges are excluded via masking.
    """
    # Get predicted bridge values (0, 1, or 2)
    # logits shape: [num_edges, 3] where classes are [0, 1, 2]
    probs = F.softmax(logits, dim=-1)  # [num_edges, 3]
    
    # Expected bridge value: E[label] = 0*P(0) + 1*P(1) + 2*P(2)
    # This gives soft continuous values during training
    bridge_values = probs[:, 0] * 0.0 + probs[:, 1] * 1.0 + probs[:, 2] * 2.0  # [num_edges]
    
    # Only consider original puzzle edges (not meta or conflict edges)
    bridge_values_masked = bridge_values * edge_mask.float()
    
    # Sum bridge values per node using scatter_add
    # We sum over edges where the node is the SOURCE (edge_index[0])
    # Note: For bidirectional graphs, each undirected edge contributes to BOTH endpoints
    src_nodes = edge_index[0]  # [num_edges]
    
    # Scatter: For each node, sum the bridge values of edges originating from it
    predicted_degrees = scatter(bridge_values_masked, src_nodes, dim=0, reduce='sum')  # [num_nodes]
    
    # Target degrees are the node capacities (first feature in x)
    target_degrees = node_capacities.float()  # [num_nodes]

    # Only puzzle nodes (n ∈ {1-8}) have degree constraints.
    # Meta nodes (n = 9,10) should not contribute to degree loss.
    # Set their target degrees to 0 so they contribute 0 loss.
    is_puzzle_node = (node_capacities >= 1) & (node_capacities <= 8)
    target_degrees = target_degrees * is_puzzle_node.float()
    
    # Ensure dimensions match (in case some nodes have no edges)
    if predicted_degrees.size(0) < target_degrees.size(0):
        padding = torch.zeros(
            target_degrees.size(0) - predicted_degrees.size(0),
            device=predicted_degrees.device,
            dtype=predicted_degrees.dtype
        )
        predicted_degrees = torch.cat([predicted_degrees, padding])
    
    # MSE loss between predicted and target degrees
    loss = F.mse_loss(predicted_degrees, target_degrees, reduction=reduction)
    
    return loss


def compute_crossing_loss(
    logits: torch.Tensor,
    edge_conflicts: Optional[List[Tuple[int, int]]],
    edge_mask: torch.Tensor,
    reduction: str = 'mean',
    mode: str = 'multiplicative'
) -> torch.Tensor:
    """
    Bridge Crossing Loss: Enforces that geometrically intersecting edges are mutually exclusive.
    
    If two edges cross in the 2D grid, at most one can have a bridge. This loss
    penalizes the model for predicting high probabilities for both crossing edges.
    
    Args:
        logits: Model output logits [num_edges, num_classes] for ALL edges (original + meta)
        edge_conflicts: List of (edge_idx1, edge_idx2) tuples for crossing edge pairs
        edge_mask: Boolean mask [num_edges] indicating original puzzle edges (excludes meta/conflict)
        reduction: 'mean' or 'sum' for loss aggregation
        mode: Loss formulation mode
            - 'multiplicative': P(e1_active) * P(e2_active) [recommended]
            - 'max_product': max(logit(e1)) * max(logit(e2)) [alternative]
    
    Returns:
        torch.Tensor: Scalar loss value
        
    Note: Expects edge_conflicts to be pre-normalized to (int, int) tuples by the
          collate function. Uses vectorized operations for efficiency.
    """
    if edge_conflicts is None or len(edge_conflicts) == 0:
        return torch.tensor(0.0, device=logits.device)
    
    # Build index tensors for vectorized gather (conflicts are pre-normalized by collate_fn)
    e1_indices = torch.tensor([c[0] for c in edge_conflicts], dtype=torch.long, device=logits.device)
    e2_indices = torch.tensor([c[1] for c in edge_conflicts], dtype=torch.long, device=logits.device)
    
    # Get probabilities for each edge having a bridge (label 1 or 2)
    probs = F.softmax(logits, dim=-1)  # [num_edges, 3]
    bridge_exists_prob = probs[:, 1] + probs[:, 2]  # [num_edges]
    
    # Vectorized gather of probabilities for all conflict pairs
    prob1 = bridge_exists_prob[e1_indices]  # [num_conflicts]
    prob2 = bridge_exists_prob[e2_indices]  # [num_conflicts]
    
    if mode == 'multiplicative':
        # Soft constraint: penalize both being active
        crossing_losses = prob1 * prob2
    elif mode == 'max_product':
        # Alternative: Use max logits (confidence-based)
        max_logit1 = logits[e1_indices].max(dim=-1).values
        max_logit2 = logits[e2_indices].max(dim=-1).values
        crossing_losses = F.relu(max_logit1) * F.relu(max_logit2)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if reduction == 'mean':
        return crossing_losses.mean()
    elif reduction == 'sum':
        return crossing_losses.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_verification_loss(
    verify_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    targets: torch.Tensor,
    edge_mask: torch.Tensor,
    edge_batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute verification loss based on whether predictions match ground truth.

    The verification target is 1.0 if ALL edges in a puzzle are correctly predicted,
    0.0 otherwise. Uses vectorized scatter operations for efficiency.

    Args:
        verify_logits: Verification logits from meta nodes [batch_size, 1]
        edge_logits: Edge prediction logits [num_edges, 3] for ALL edges
        targets: Ground truth edge labels [num_original_edges]
        edge_mask: Boolean mask [num_edges] indicating original puzzle edges
        edge_batch: Batch index for each edge [num_edges]

    Returns:
        loss: BCE loss for verification
        balanced_acc: Balanced verification accuracy (average of positive and negative recall)
        recall_pos: Recall for positive class (perfect puzzles)
        recall_neg: Recall for negative class (imperfect puzzles)
    """
    if verify_logits is None or verify_logits.numel() == 0:
        device = edge_logits.device
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
    # Get predictions for original edges
    edge_preds = edge_logits[edge_mask].argmax(dim=-1)
    edge_batch_original = edge_batch[edge_mask]
    
    # Vectorized per-puzzle correctness using scatter
    # Per-edge incorrectness: 1 if wrong, 0 if correct
    edge_incorrect = (edge_preds != targets).long()
    
    # Sum incorrectness per puzzle - if sum == 0, puzzle is perfect
    num_puzzles = edge_batch_original.max().item() + 1
    errors_per_puzzle = scatter(edge_incorrect, edge_batch_original, dim=0, 
                                 dim_size=num_puzzles, reduce='sum')
    
    # Perfect if no errors (errors_per_puzzle == 0)
    verify_targets = (errors_per_puzzle == 0).float().unsqueeze(-1)  # [batch_size, 1]
    
    # Calculate dynamic class weight for the positive class (Perfect puzzles)
    # Using Laplace smoothing (add 1 to numerator and denominator) for stability
    num_pos = verify_targets.sum()
    
    # Scale the positive class by the ratio of negatives to positives
    # weight = (num_neg + 1) / (num_pos + 1)
    pos_weight = (float(num_puzzles) - num_pos + 1.0) / (num_pos + 1.0)
    
    # BCE loss with logits and dynamic positive weight
    loss = F.binary_cross_entropy_with_logits(verify_logits, verify_targets, pos_weight=pos_weight)
    
    # Compute balanced accuracy (average of recall for each class)
    verify_preds = (torch.sigmoid(verify_logits) > 0.5).float()
    
    pos_mask = (verify_targets == 1.0)
    neg_mask = (verify_targets == 0.0)
    
    num_pos = pos_mask.sum()
    num_neg = neg_mask.sum()
    
    recall_pos = (verify_preds[pos_mask] == 1.0).float().mean() if num_pos > 0 else torch.tensor(1.0, device=verify_logits.device)
    recall_neg = (verify_preds[neg_mask] == 0.0).float().mean() if num_neg > 0 else torch.tensor(1.0, device=verify_logits.device)
    
    balanced_acc = (recall_pos + recall_neg) / 2.0

    return loss, balanced_acc, recall_pos, recall_neg


def compute_combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    edge_index: torch.Tensor,
    node_capacities: torch.Tensor,
    edge_conflicts: Optional[List[Tuple[int, int]]],
    edge_mask: torch.Tensor,
    loss_weights: Optional[Dict[str, float]] = None,
    verify_logits: Optional[torch.Tensor] = None,
    edge_batch: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Combined loss function with classification + auxiliary losses + verification.
    
    Args:
        logits: Model output logits [num_edges, num_classes] for ALL edges
        targets: Ground truth edge labels [num_original_edges] for original edges only
        edge_index: Graph connectivity [2, num_edges]
        node_capacities: Island capacity values [num_nodes]
        edge_conflicts: List of (edge_idx1, edge_idx2) tuples for crossing edges
        edge_mask: Boolean mask [num_edges] indicating original puzzle edges
        loss_weights: Dict with keys 'ce', 'degree', 'crossing', 'verify' for loss weighting
        verify_logits: Optional verification logits from meta nodes [batch_size, 1]
        edge_batch: Optional batch index for each edge [num_edges] (required if verify_logits provided)
    
    Returns:
        dict: Contains 'total', 'ce', 'degree', 'crossing', 'verify', 'verify_acc' values
    """
    if loss_weights is None:
        loss_weights = {
            'ce': 1.0,
            'degree': 0.1,
            'crossing': 0.5,
            'verify': 0.0
        }
    
    # 1. Standard Cross-Entropy Loss (classification on original edges only)
    logits_original = logits[edge_mask]
    loss_ce = F.cross_entropy(logits_original, targets, reduction='mean')
    
    # 2. Degree Violation Loss (island counting constraint)
    # This operates on ALL edges but masks appropriately inside
    loss_degree = compute_degree_violation_loss(
        logits, edge_index, node_capacities, edge_mask, reduction='mean'
    )
    
    # 3. Bridge Crossing Loss (mutual exclusion constraint)
    # This operates on ALL edges but masks appropriately inside
    loss_crossing = compute_crossing_loss(
        logits, edge_conflicts, edge_mask, reduction='mean', mode='multiplicative'
    )
    
    # 4. Verification Loss (self-critique)
    verify_weight = loss_weights.get('verify', 0.0)
    if verify_logits is not None and edge_batch is not None and verify_weight > 0:
        loss_verify, verify_acc, verify_recall_pos, verify_recall_neg = compute_verification_loss(
            verify_logits, logits, targets, edge_mask, edge_batch
        )
    else:
        loss_verify = torch.tensor(0.0, device=logits.device)
        verify_acc = torch.tensor(0.0, device=logits.device)
        verify_recall_pos = torch.tensor(0.0, device=logits.device)
        verify_recall_neg = torch.tensor(0.0, device=logits.device)
    
    # Weighted combination
    total_loss = (
        loss_weights['ce'] * loss_ce +
        loss_weights.get('degree', 0.0) * loss_degree +
        loss_weights.get('crossing', 0.0) * loss_crossing +
        verify_weight * loss_verify
    )
    
    return {
        'total': total_loss,
        'ce': loss_ce,
        'degree': loss_degree,
        'crossing': loss_crossing,
        'verify': loss_verify,
        'verify_acc': verify_acc,
        'verify_recall_pos': verify_recall_pos,
        'verify_recall_neg': verify_recall_neg
    }


def asymmetric_mse_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha: float = 5.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Asymmetric MSE Loss for AR safety.

    Penalizes overshooting (predicting too many bridges) much more heavily than
    undershooting. This encourages conservative behavior.

    Args:
        y_pred: Predicted values [batch_size]
        y_true: Target values [batch_size]
        alpha: Penalty multiplier for overshooting (default: 5.0)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss value
    """
    diff = y_pred - y_true

    # Overshoot: penalty * (diff)^2 when diff > 0
    # Undershoot: (diff)^2 when diff <= 0
    loss = torch.where(
        diff > 0,
        alpha * diff**2,  # Overshoot penalty
        diff**2           # Undershoot penalty
    )

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def ordinal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Ordinal BCE Loss for conditional action head.

    Targets should be integer values 0, 1, or 2.
    Converts to binary targets and computes BCE.

    Args:
        logits: Model logits [batch_size, 2] (p_ge1, p_ge2)
        targets: Target actions [batch_size] (0, 1, or 2)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss value
    """
    # Convert targets to binary format
    # Target 0: [0, 0] (not >=1, not >=2)
    # Target 1: [1, 0] (>=1 but not >=2)
    # Target 2: [1, 1] (>=1 and >=2)
    binary_targets = torch.zeros_like(logits)

    binary_targets[targets >= 1, 0] = 1.0  # p_ge1
    binary_targets[targets >= 2, 1] = 1.0  # p_ge2

    # Binary cross entropy with logits
    loss = F.binary_cross_entropy_with_logits(
        logits, binary_targets, reduction=reduction
    )

    return loss


def compute_ar_loss(
    output: torch.Tensor,
    targets: torch.Tensor,
    current_bridges: torch.Tensor,
    edge_mask: torch.Tensor,
    head_type: str = 'regression',
    overshoot_penalty: float = 5.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute AR loss based on head type.

    Args:
        output: Model output (shape depends on head_type)
        targets: Target remaining bridges [num_edges]
        current_bridges: Current bridge counts [num_edges]
        edge_mask: Boolean mask for valid edges [num_edges]
        head_type: 'regression' or 'conditional'
        overshoot_penalty: Penalty for overshooting (regression only)
        reduction: Loss reduction method

    Returns:
        Scalar loss value
    """
    # Only compute loss on valid edges (not masked or locked)
    valid_mask = edge_mask & (current_bridges < 2)  # Not locked (max 2 bridges)
    if not valid_mask.any():
        return torch.tensor(0.0, device=output.device)

    targets_masked = targets[valid_mask]

    if head_type == 'regression':
        # Regression: output is [num_edges] continuous values
        output_masked = output[valid_mask]
        return asymmetric_mse_loss(
            output_masked, targets_masked,
            alpha=overshoot_penalty, reduction=reduction
        )

    elif head_type == 'conditional':
        # Conditional: output is [num_edges, 2] logits
        output_masked = output[valid_mask]  # [num_valid_edges, 2]
        targets_masked_int = targets_masked.long()
        return ordinal_bce_loss(
            output_masked, targets_masked_int, reduction=reduction
        )

    else:
        raise ValueError(f"Unknown head_type: {head_type}")

