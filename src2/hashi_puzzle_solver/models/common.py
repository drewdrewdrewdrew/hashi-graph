"""Common model components and utilities."""
import torch
from torch.nn import Dropout, LayerNorm, Linear, ReLU, Sequential


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str = "relu",
    dropout: float = 0.0,
    use_layer_norm: bool = False,
) -> Sequential:
    """
    Standard MLP builder with optional normalization and dropout.
    
    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        hidden_dim: Dimension of hidden layers.
        num_layers: Number of hidden layers.
        activation: Activation function name ("relu" supported).
        dropout: Dropout probability.
        use_layer_norm: Whether to use LayerNorm before activation.
        
    Returns:
        Sequential model.
    """
    layers = []
    curr_dim = input_dim
    
    # Activation mapping
    act_cls = ReLU
    if activation.lower() != "relu":
        raise ValueError(f"Unsupported activation: {activation}")
    
    for _ in range(num_layers):
        layers.append(Linear(curr_dim, hidden_dim))
        if use_layer_norm:
            layers.append(LayerNorm(hidden_dim))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(Dropout(dropout))
        curr_dim = hidden_dim
        
    layers.append(Linear(curr_dim, output_dim))
    return Sequential(*layers)
