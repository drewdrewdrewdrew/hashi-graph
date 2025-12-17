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
        
    Implementation:
        For each pair of crossing edges (e1, e2):
            prob_e1_active = P(label=1) + P(label=2)  [probability bridge exists]
            prob_e2_active = P(label=1) + P(label=2)
            
            loss_pair = prob_e1_active * prob_e2_active
        
        Total loss = mean(loss_pair) over all crossing pairs
        
        Why multiplicative: If both edges have high probability, the product explodes.
        Gradient pushes at least one probability toward zero (mutual exclusion).
        
    Note: Only original puzzle edges (edge_mask=True) are considered for conflicts.
          Meta and conflict edges are excluded via masking.
    """
    if edge_conflicts is None or len(edge_conflicts) == 0:
        # No crossing constraints to enforce
        return torch.tensor(0.0, device=logits.device)
    
    # Get probabilities for each edge having a bridge (label 1 or 2)
    probs = F.softmax(logits, dim=-1)  # [num_edges, 3]
    
    # Probability that a bridge exists: P(label=1) + P(label=2)
    bridge_exists_prob = probs[:, 1] + probs[:, 2]  # [num_edges]
    
    # Only consider original puzzle edges
    bridge_exists_prob = bridge_exists_prob * edge_mask.float()
    
    # Compute loss for each crossing pair
    crossing_losses = []
    
    for conflict in edge_conflicts:
        if not conflict or len(conflict) != 2:
            raise ValueError(f"Edge conflict entry {conflict} is malformed.")

        edge_idx1, edge_idx2 = conflict

        while isinstance(edge_idx1, (list, tuple)) and len(edge_idx1) > 0:
            edge_idx1 = edge_idx1[0]
        while isinstance(edge_idx2, (list, tuple)) and len(edge_idx2) > 0:
            edge_idx2 = edge_idx2[0]

        if hasattr(edge_idx1, 'item'):
            edge_idx1 = edge_idx1.item()
        if hasattr(edge_idx2, 'item'):
            edge_idx2 = edge_idx2.item()

        try:
            edge_idx1 = int(edge_idx1)
            edge_idx2 = int(edge_idx2)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Edge conflict {conflict} contains non-integer indices."
            ) from exc

        num_edges = edge_mask.size(0)
        if not (0 <= edge_idx1 < num_edges and 0 <= edge_idx2 < num_edges):
            raise ValueError(
                f"Edge conflict {conflict} references indices outside [0, {num_edges})."
            )

        if not (edge_mask[edge_idx1] and edge_mask[edge_idx2]):
            raise ValueError(
                f"Edge conflict {conflict} refers to non-original puzzle edges."
            )
        
        prob1 = bridge_exists_prob[edge_idx1]
        prob2 = bridge_exists_prob[edge_idx2]
        
        if mode == 'multiplicative':
            # Soft constraint: penalize both being active
            pair_loss = prob1 * prob2
        elif mode == 'max_product':
            # Alternative: Use max logits (confidence-based)
            max_logit1 = logits[edge_idx1].max()
            max_logit2 = logits[edge_idx2].max()
            pair_loss = F.relu(max_logit1) * F.relu(max_logit2)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        crossing_losses.append(pair_loss)
    
    if len(crossing_losses) == 0:
        return torch.tensor(0.0, device=logits.device)
    
    # Aggregate losses
    crossing_losses = torch.stack(crossing_losses)
    
    if reduction == 'mean':
        return crossing_losses.mean()
    elif reduction == 'sum':
        return crossing_losses.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    edge_index: torch.Tensor,
    node_capacities: torch.Tensor,
    edge_conflicts: Optional[List[Tuple[int, int]]],
    edge_mask: torch.Tensor,
    loss_weights: Optional[Dict[str, float]] = None
) -> Dict[str, torch.Tensor]:
    """
    Combined loss function with classification + auxiliary losses.
    
    Args:
        logits: Model output logits [num_edges, num_classes] for ALL edges
        targets: Ground truth edge labels [num_original_edges] for original edges only
        edge_index: Graph connectivity [2, num_edges]
        node_capacities: Island capacity values [num_nodes]
        edge_conflicts: List of (edge_idx1, edge_idx2) tuples for crossing edges
        edge_mask: Boolean mask [num_edges] indicating original puzzle edges
        loss_weights: Dict with keys 'ce', 'degree', 'crossing' for loss weighting
    
    Returns:
        dict: Contains 'total', 'ce', 'degree', 'crossing' loss values
    """
    if loss_weights is None:
        loss_weights = {
            'ce': 1.0,
            'degree': 0.1,
            'crossing': 0.5
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
    
    # Weighted combination
    total_loss = (
        loss_weights['ce'] * loss_ce +
        loss_weights['degree'] * loss_degree +
        loss_weights['crossing'] * loss_crossing
    )
    
    return {
        'total': total_loss,
        'ce': loss_ce,
        'degree': loss_degree,
        'crossing': loss_crossing
    }

