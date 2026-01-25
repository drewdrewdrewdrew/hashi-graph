"""Loss calculator for Hashi Puzzle Solver."""

import torch
import torch.nn.functional as func
from ..models.config import HashiModelConfig
from .degree import DegreeLoss
from .crossing import CrossingLoss
from .verification import VerificationLoss


class HashiLossCalculator:
    """
    Orchestrates multiple loss components based on configuration.
    """

    def __init__(self, config: HashiModelConfig):
        self.config = config
        self.loss_weights = config.training.loss_weights
        
        self.degree_loss = DegreeLoss()
        self.crossing_loss = CrossingLoss()
        self.verification_loss = VerificationLoss()

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        edge_index: torch.Tensor,
        node_capacities: torch.Tensor,
        edge_conflict_index: torch.Tensor | None,
        edge_mask: torch.Tensor,
        verify_logits: torch.Tensor | None = None,
        edge_batch: torch.Tensor | None = None,
        velocity_target: torch.Tensor | None = None,
        aux_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined weighted loss."""
        if aux_logits is None:
            aux_logits = logits

        # 1. Base Loss (classification OR flow matching)
        if velocity_target is not None:
            loss_ce = func.mse_loss(logits, velocity_target, reduction="mean")
        else:
            logits_original = logits[edge_mask]
            targets_original = targets[edge_mask]
            loss_ce = func.cross_entropy(logits_original, targets_original, reduction="mean")

        # 2. Auxiliary Losses
        loss_degree = self.degree_loss(
            logits=aux_logits,
            edge_index=edge_index,
            node_capacities=node_capacities,
            edge_mask=edge_mask
        )
        
        loss_crossing = self.crossing_loss(
            logits=aux_logits,
            edge_conflict_index=edge_conflict_index
        )

        # 3. Verification Loss
        verify_weight = getattr(self.loss_weights, "verify", 0.0)
        if verify_logits is not None and edge_batch is not None and verify_weight > 0:
            loss_verify, verify_acc, verify_recall_pos, verify_recall_neg = self.verification_loss(
                verify_logits=verify_logits,
                edge_logits=aux_logits,
                targets=targets,
                edge_mask=edge_mask,
                edge_batch=edge_batch
            )
        else:
            loss_verify = torch.tensor(0.0, device=logits.device)
            verify_acc = torch.tensor(0.0, device=logits.device)
            verify_recall_pos = torch.tensor(0.0, device=logits.device)
            verify_recall_neg = torch.tensor(0.0, device=logits.device)

        # 4. Weighted Combination
        total_loss = (
            self.loss_weights.ce * loss_ce +
            self.loss_weights.degree * loss_degree +
            self.loss_weights.crossing * loss_crossing +
            verify_weight * loss_verify
        )

        return {
            "total": total_loss,
            "ce": loss_ce,
            "degree": loss_degree,
            "crossing": loss_crossing,
            "verify": loss_verify,
            "verify_acc": verify_acc,
            "verify_recall_pos": verify_recall_pos,
            "verify_recall_neg": verify_recall_neg,
        }
