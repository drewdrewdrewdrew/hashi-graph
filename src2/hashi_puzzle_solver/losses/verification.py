"""Verification loss for Hashi Puzzle Solver."""

import torch
import torch.nn.functional as func
from torch_geometric.utils import scatter


class VerificationLoss(torch.nn.Module):
    """
    Computes verification loss by comparing global puzzle validity predictions 
    with actual prediction errors.
    """

    def forward(
        self,
        verify_logits: torch.Tensor | None,
        edge_logits: torch.Tensor,
        targets: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute verification loss.
        
        Args:
            verify_logits: Logits from verification head [num_graphs, 1].
            edge_logits: Logits for edge bridge counts [num_edges, 3].
            targets: Target bridge counts [num_edges].
            edge_mask: Mask for puzzle edges.
            edge_batch: Batch indices for edges.
            
        Returns:
            Tuple of (loss, balanced_acc, recall_pos, recall_neg).
        """
        if verify_logits is None or verify_logits.numel() == 0:
            device = edge_logits.device
            return (
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
            )

        # 1. Determine ground truth validity for each puzzle in batch
        # A puzzle is valid if ALL of its puzzle edges are correctly predicted
        edge_preds = edge_logits[edge_mask].argmax(dim=-1)
        targets_puzzle = targets[edge_mask]
        edge_batch_puzzle = edge_batch[edge_mask]
        
        edge_incorrect = (edge_preds != targets_puzzle).long()
        num_puzzles = verify_logits.size(0)
        
        errors_per_puzzle = scatter(
            edge_incorrect, edge_batch_puzzle, dim=0, dim_size=num_puzzles, reduce="sum"
        )
        
        # Target: 1.0 if correct (0 errors), 0.0 otherwise
        verify_targets = (errors_per_puzzle == 0).float().unsqueeze(-1)
        
        # 2. Weighted BCE Loss to handle imbalance (most puzzles are incorrect early on)
        num_pos = verify_targets.sum()
        pos_weight = (float(num_puzzles) - num_pos + 1.0) / (num_pos + 1.0)
        
        loss = func.binary_cross_entropy_with_logits(
            verify_logits, verify_targets, pos_weight=pos_weight
        )
        
        # 3. Metrics
        with torch.no_grad():
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
