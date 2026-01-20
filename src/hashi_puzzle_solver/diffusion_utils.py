"""Utilities for Denoising Diffusion in Hashi GNN."""

import torch
from torch_geometric.data import Data

from .train_utils import update_node_features


def inject_noise(
    data: Data,
    noise_rate: float,
    bridge_label_idx: int,
    is_labeled_idx: int,
    model_config: dict,
    device: torch.device,
) -> Data:
    """Inject discrete noise (flips) for diff-discrete mode."""
    # Clone data to avoid in-place changes to the dataset
    data = data.clone()

    # Original edges only
    if not hasattr(data, "edge_mask") or data.edge_mask is None:
        return data

    original_edge_indices = torch.where(data.edge_mask)[0]
    num_edges = len(original_edge_indices)

    current_bridges = data.y.clone().float()

    num_to_corrupt = int(num_edges * noise_rate)
    if num_to_corrupt > 0:
        perm = torch.randperm(num_edges, device=device)[:num_to_corrupt]
        corrupt_indices = original_edge_indices[perm]

        # Randomly sample labels from {0, 1, 2} but DIFFERENT from ground truth
        # To do this simply: current + rand(1, 3) mod 3
        shifts = torch.randint(1, 3, (num_to_corrupt,), device=device).float()
        current_bridges[corrupt_indices] = (
            current_bridges[corrupt_indices] + shifts
        ) % 3

    # Update edge_attr with these noisy labels
    if data.edge_attr is not None:
        if bridge_label_idx is not None:
            data.edge_attr[:, bridge_label_idx] = current_bridges
        if is_labeled_idx is not None:
            data.edge_attr[:, is_labeled_idx] = 1.0

    # Update node features (unused_capacity)
    if model_config.get("use_unused_capacity", True):
        data.x = update_node_features(
            data.x,
            current_bridges,
            data.edge_index,
            data.node_type,
            model_config
        )

    return data


def inject_continuous_noise(
    data: Data,
    alpha: float | torch.Tensor,
    sigma: float | torch.Tensor,
    scale: float | torch.Tensor,
    bridge_logits_idx: int,
    model_config: dict,
    device: torch.device,
) -> Data:
    """
    Inject continuous logit noise for diff-cont mode.

    Follows the "Centered Logit" recipe:
    1. One-Hot
    2. Centering (y - 1/3)
    3. Target Construction (y_centered * scale)
    4. Signal Interpolation (y_target * alpha)
    5. Noise Injection (signal + N(0, sigma))

    Returns Data with x_input in edge_attr and updated node capacity.
    Supports both scalar and per-graph tensor inputs for alpha, sigma, and scale.
    """
    data = data.clone()
    num_graphs = getattr(data, "num_graphs", 1)

    # 1. One-Hot
    # Handle cases where data.y might be shorter than edge_index (e.g. meta-edges)
    num_edges = data.edge_index.size(1)
    num_labels = data.y.size(0)

    y_onehot_full = torch.zeros((num_edges, 3), device=data.y.device)
    # Meta-edges (if any) default to label 0 (background)
    if num_labels > 0:
        y_onehot_full[:num_labels] = torch.nn.functional.one_hot(
            data.y.long(), num_classes=3
        ).float()

    # 2. Centering
    y_centered = y_onehot_full - (1.0 / 3.0)

    # Prepare broadcasting if tensors are provided
    batch_attr = getattr(data, "batch", None)
    if batch_attr is None:
        batch_attr = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    edge_batch = batch_attr[data.edge_index[0]]

    def expand_param(p: float | torch.Tensor, name: str) -> float | torch.Tensor:
        if isinstance(p, torch.Tensor):
            if p.dim() == 0:
                return p.view(1, 1)
            if p.size(0) != num_graphs:
                msg = (
                    f"Param {name} tensor size {p.size(0)} "
                    f"!= num_graphs {num_graphs}"
                )
                raise ValueError(msg)
            return p[edge_batch].view(-1, 1)
        return p

    alpha_edges = expand_param(alpha, "alpha")
    sigma_edges = expand_param(sigma, "sigma")
    scale_edges = expand_param(scale, "scale")

    # 3. Target Construction
    y_target = y_centered * scale_edges

    # 4. Signal Interpolation
    x_signal = y_target * alpha_edges

    # 5. Noise Injection
    noise = torch.randn_like(x_signal) * sigma_edges
    x_input = x_signal + noise

    # 6. Edge Update
    if data.edge_attr is not None and bridge_logits_idx is not None:
        # bridge_logits_idx is the start of a 3-wide block
        data.edge_attr[:, bridge_logits_idx:bridge_logits_idx + 3] = x_input

    # 7. Constraint Update (Sharp Prop)
    if model_config.get("use_unused_capacity", True):
        # Calculate current labels based on argmax of noisy logits
        current_labels = x_input.argmax(dim=-1).float()
        data.x = update_node_features(
            data.x,
            current_labels,
            data.edge_index,
            data.node_type,
            model_config
        )

    return data


def estimate_signal_noise_stats(
    logits: torch.Tensor,
    y: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
    scale: float | torch.Tensor = 8.0,
) -> torch.Tensor:
    """
    Estimate the "actual" alpha and sigma from model outputs for ground truth targets.

    Used for the Prophet Head (f.7) to predict the noise level of the board
    when aux_predict_output_noise is enabled.

    Logic:
    1. Center logits and targets.
    2. Estimate alpha via projection of logits onto centered targets.
    3. Estimate sigma via residual std between logits and projected signal.
    """
    # 1. Center targets and logits
    num_labels = y.size(0)
    # Slice logits to match targets if necessary (handles dynamic meta-edges)
    logits = logits[:num_labels]
    edge_batch = edge_batch[:num_labels]

    y_onehot = torch.nn.functional.one_hot(y.long(), num_classes=3).float()
    y_centered = y_onehot - (1.0 / 3.0)

    # We assume targets are scaled by 'scale' (target signal strength)
    if isinstance(scale, torch.Tensor):
        scale_expanded = scale[edge_batch].view(-1, 1) if scale.dim() == 1 else scale
    else:
        scale_expanded = scale

    y_target = y_centered * scale_expanded

    # 2. Estimate Alpha via Projection
    # We do this per-graph
    dot_logits_y = (logits * y_target).sum(dim=-1)
    dot_y_y = (y_target * y_target).sum(dim=-1)

    # Pool dots per graph
    graph_dot_logits_y = torch.zeros(
        num_graphs, device=logits.device
    ).scatter_add_(0, edge_batch, dot_logits_y)
    graph_dot_y_y = torch.zeros(
        num_graphs, device=logits.device
    ).scatter_add_(0, edge_batch, dot_y_y)

    estimated_alphas = graph_dot_logits_y / (graph_dot_y_y + 1e-9)

    # 3. Estimate Sigma via Residual
    alpha_expanded = estimated_alphas[edge_batch].view(-1, 1)
    residual = logits - (alpha_expanded * y_target)

    # Sigma = sqrt(mean(residual^2)) per graph
    residual_sq_mean = (residual * residual).mean(dim=-1)
    graph_residual_sq_sum = torch.zeros(
        num_graphs, device=logits.device
    ).scatter_add_(0, edge_batch, residual_sq_mean)

    # Count edges per graph for averaging
    edge_counts = torch.zeros(
        num_graphs, device=logits.device
    ).scatter_add_(0, edge_batch, torch.ones_like(edge_batch).float())

    estimated_sigmas = torch.sqrt(graph_residual_sq_sum / (edge_counts + 1e-9) + 1e-9)

    return torch.stack([estimated_sigmas, estimated_alphas], dim=-1)


def inject_flow_noise(
    data: Data,
    t: torch.Tensor,
    bridge_logits_idx: int,
    model_config: dict,
    training_config: dict,
    device: torch.device,
) -> Data:
    """
    Inject flow-matching noise for flow-blind mode.

    Follows linear interpolation: x_t = (1-t)*noise + t*clean
    Velocity target = clean - noise

    Args:
        data: Data object
        t: Time tensor [num_graphs, 1]
        bridge_logits_idx: Start index of bridge logits in edge_attr
        model_config: Model configuration
        training_config: Training configuration (for scales/noise levels)
        device: Torch device

    Returns:
        Modified Data object with x_t in edge_attr and 'velocity_target' attribute.
    """
    data = data.clone()
    num_graphs = getattr(data, "num_graphs", 1)

    # 1. Prepare Clean State (y_target)
    # Handle cases where data.y might be shorter than edge_index (e.g. meta-edges)
    num_edges = data.edge_index.size(1)
    num_labels = data.y.size(0)

    y_onehot_full = torch.zeros((num_edges, 3), device=data.y.device)
    if num_labels > 0:
        y_onehot_full[:num_labels] = torch.nn.functional.one_hot(
            data.y.long(), num_classes=3
        ).float()

    # 2. Prepare Centered State
    y_centered = y_onehot_full - (1.0 / 3.0)

    # Use scale_max from config, default to 8.0
    clean_scale = training_config.get("scale_max", 8.0)
    y_clean = y_centered * clean_scale

    # 2. Prepare Noise State
    # Use sigma_max from config, default to 2.0
    noise_sigma = training_config.get("sigma_max", 2.0)
    noise = torch.randn_like(y_clean) * noise_sigma

    # 3. Interpolate x_t = (1-t)*noise + t*clean
    batch_attr = getattr(data, "batch", None)
    if batch_attr is None:
        batch_attr = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    edge_batch = batch_attr[data.edge_index[0]]

    t_edges = t[edge_batch].view(-1, 1)  # [num_edges, 1]

    x_t = (1.0 - t_edges) * noise + t_edges * y_clean

    # 4. Target: Velocity = clean - noise
    velocity_target = y_clean - noise
    data.velocity_target = velocity_target

    # 5. Update edge_attr
    if data.edge_attr is not None and bridge_logits_idx is not None:
        data.edge_attr[:, bridge_logits_idx:bridge_logits_idx + 3] = x_t

    # 6. Update node features (unused_capacity)
    if model_config.get("use_unused_capacity", True):
        current_labels = x_t.argmax(dim=-1).float()
        data.x = update_node_features(
            data.x,
            current_labels,
            data.edge_index,
            data.node_type,
            model_config
        )

    return data
