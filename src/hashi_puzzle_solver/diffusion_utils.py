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
    Inject noise into the ground truth labels to create a corrupted input state.
    
    Args:
        data: PyG Data object.
        noise_rate: Fraction of edges to corrupt (0.0 to 1.0).
        bridge_label_idx: Index of bridge label feature in edge_attr.
        is_labeled_idx: Index of is_labeled feature in edge_attr.
        model_config: Model configuration.
        device: Torch device.
        
    Returns:
        Data: PyG Data object with corrupted edge_attr and updated node features.
    """
    if noise_rate <= 0.0:
        # Start from empty board if noise rate is 0 (Wait, 0 noise means perfect labels?)
        # The plan says: "Initial state: empty board" for inference.
        # For training: "take ground truth Y and randomly flip K% of edges".
        # So noise_rate 0 means X = Y.
        # But wait, high noise means more corruption. 
        # If noise_rate=1.0, all edges are corrupted?
        # Let's follow the "flip K% of edges" instruction.
        pass

    # Clone data to avoid in-place changes to the dataset
    data = data.clone()
    
    # Original edges only
    if not hasattr(data, "edge_mask"):
        return data
        
    original_edge_indices = torch.where(data.edge_mask)[0]
    num_edges = len(original_edge_indices)
    
    # Start with ground truth as the base for X
    # Actually, the plan says "take ground truth Y and randomly flip K% of edges".
    # This means X = Y initially, then modify K% of X.
    
    current_bridges = data.y.clone().float()
    
    num_to_corrupt = int(num_edges * noise_rate)
    if num_to_corrupt > 0:
        perm = torch.randperm(num_edges, device=device)[:num_to_corrupt]
        corrupt_indices = original_edge_indices[perm]
        
        # Randomly sample labels from {0, 1, 2}
        random_labels = torch.randint(0, 3, (num_to_corrupt,), device=device).float()
        current_bridges[corrupt_indices] = random_labels
            
    # Update edge_attr with these noisy labels
    if data.edge_attr is not None:
        data.edge_attr[:, bridge_label_idx] = current_bridges
        data.edge_attr[:, is_labeled_idx] = 1.0 # All edges are "labeled" (with noise)
        
    # Update node features (unused_capacity)
    # We need to first RESET unused_capacity to original capacity, then subtract current_bridges
    if model_config.get("use_unused_capacity", True):
        # The data.x initially has original capacity in the unused_capacity slot 
        # (because HashiDataset initializes it that way)
        # BUT update_node_features expects it to be the ORIGINAL capacity 
        # to subtract current_bridges from.
        # Let's verify how HashiDataset initializes it.
        
        unused_idx = get_unused_capacity_index(model_config)
        
        # Reset unused_capacity to original capacity (which is in node_type/x[:, 0] usually)
        # Wait, HashiDataset sets data.x[:, unused_idx] = capacity.
        # So we can just call update_node_features directly.
        
        data.x = update_node_features(
            data.x,
            current_bridges,
            data.edge_index,
            data.node_type,
            model_config
        )
        
    return data
