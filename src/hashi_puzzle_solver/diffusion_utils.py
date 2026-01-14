"""Utilities for Denoising Diffusion in Hashi GNN."""

import torch
from torch_geometric.data import Data
from .train_utils import update_node_features, get_unused_capacity_index

def inject_noise(
    data: Data,
    noise_rate: float,
    bridge_label_idx: int,
    is_labeled_idx: int,
    model_config: dict,
    device: torch.device,
) -> Data:
    """
    Inject discrete noise (flips) for diff-discrete mode.
    """
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
        current_bridges[corrupt_indices] = (current_bridges[corrupt_indices] + shifts) % 3
            
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
    # Ground truth y is in {0, 1, 2}
    y_onehot = torch.nn.functional.one_hot(data.y.long(), num_classes=3).float()
    
    # 2. Centering
    y_centered = y_onehot - (1.0 / 3.0)
    
    # Prepare broadcasting if tensors are provided
    batch_attr = getattr(data, "batch", None)
    if batch_attr is None:
        batch_attr = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    edge_batch = batch_attr[data.edge_index[0]]

    def expand_param(p, name):
        if isinstance(p, torch.Tensor):
            if p.dim() == 0:
                return p.view(1, 1)
            if p.size(0) != num_graphs:
                raise ValueError(f"Param {name} tensor size {p.size(0)} != num_graphs {num_graphs}")
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
        data.edge_attr[:, bridge_logits_idx:bridge_logits_idx+3] = x_input
        
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
