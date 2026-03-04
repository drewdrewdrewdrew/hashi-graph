"""Tests for HashiGraphModel shell."""

import torch
from hashi_puzzle_solver.models.config import HashiModelConfig
from hashi_puzzle_solver.models.factory import ModelFactory
from hashi_puzzle_solver.models.features import NodeFeatureManager, EdgeFeatureManager
from hashi_puzzle_solver.models.encoders import NodeEncoder, EdgeEncoder
from hashi_puzzle_solver.models.backbone import GraphBackbone
from hashi_puzzle_solver.models.heads import EdgeHead, ProphetHead
from hashi_puzzle_solver.models.core import HashiGraphModel


def test_full_model_forward():
    """Test full forward pass of the integrated model shell."""
    config = HashiModelConfig()
    config.model.node_embedding_dim = 16
    config.model.hidden_channels = 32
    config.model.use_noise_head = True
    # Explicitly set some features for the test
    config.model.use_capacity = True
    config.model.use_unused_capacity = False
    config.model.use_structural_degree = False
    config.model.use_conflict_status = False
    config.model.use_closeness_centrality = False
    config.model.use_articulation_points = False
    config.model.use_spectral_features = False

    nm = NodeFeatureManager(config.model)
    em = EdgeFeatureManager(config.model)
    
    node_enc = NodeEncoder(config.model, nm)
    edge_enc = EdgeEncoder(config.model, em)
    
    # Calculate the dimension after EdgeEncoder
    encoded_edge_dim = edge_enc.output_dim
    
    backbone = GraphBackbone(
        node_input_dim=32,
        hidden_channels=32,
        num_layers=2,
        edge_dim=encoded_edge_dim,
    )
    
    edge_head = EdgeHead(config.model, 32, encoded_edge_dim)
    prophet_head = ProphetHead(config.model, 32)
    
    model = HashiGraphModel(
        config,
        node_enc,
        edge_enc,
        backbone,
        edge_head,
        prophet_head
    )
    
    # Mock data
    num_nodes = 10
    num_edges = 20
    x = torch.zeros((num_nodes, 1)) # Only capacity
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, em.num_edge_feats)
    node_type = torch.tensor([9] + [1] * (num_nodes - 1))
    
    # Forward pass
    logits = model(x, edge_index, edge_attr=edge_attr, node_type=node_type)
    assert logits.shape == (num_edges, 3)
    
    # Forward pass with noise head
    logits, noise_logits = model(
        x, 
        edge_index, 
        edge_attr=edge_attr, 
        node_type=node_type,
        return_noise=True
    )
    assert logits.shape == (num_edges, 3)
    assert noise_logits.shape == (1, 2)


def _make_minimal_model_config(use_noise_head: bool = True) -> HashiModelConfig:
    """Return a minimal HashiModelConfig suitable for unit tests."""
    config = HashiModelConfig()
    config.model.node_embedding_dim = 16
    config.model.hidden_channels = 32
    config.model.num_layers = 2
    config.model.heads = 2
    config.model.use_noise_head = use_noise_head
    # Disable all optional node features to keep input width minimal
    config.model.use_capacity = True
    config.model.use_unused_capacity = False
    config.model.use_structural_degree = False
    config.model.use_conflict_status = False
    config.model.use_closeness_centrality = False
    config.model.use_articulation_points = False
    config.model.use_spectral_features = False
    # Disable optional edge features
    config.model.use_categorical_edge_types = False
    config.model.use_continuous_edge_labels = False
    config.model.use_cut_edges = False
    config.model.use_potential_crossing = False
    config.model.use_conflict_edges = False
    config.model.use_row_col_meta = False
    config.model.use_meta_mesh = False
    config.model.use_meta_row_col_edges = False
    config.model.use_component_meta = False
    config.model.use_noise_in_message_passing = False
    config.model.use_noise_in_prediction = False
    config.model.use_noise_in_global_meta = False
    return config


def test_no_noise_head_returns_single_tensor():
    """Model with use_noise_head=False returns only edge_logits, never a tuple.

    Verifies Fix 4: setting use_noise_head=False is safe — no prophet_head is
    created and return_noise=True is a no-op (no crash, no unexpected tuple).
    """
    config = _make_minimal_model_config(use_noise_head=False)
    device = torch.device("cpu")
    model = ModelFactory.create_model(config, device)

    # prophet_head must be None when use_noise_head=False
    assert model.prophet_head is None, "prophet_head should be None when use_noise_head=False"

    em = EdgeFeatureManager(config.model)
    num_nodes, num_edges = 8, 12
    x = torch.zeros(num_nodes, 1)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, em.num_edge_feats)
    node_type = torch.tensor([9] + [1] * (num_nodes - 1))

    # return_noise=True should be silently ignored since prophet_head is None
    result = model(
        x,
        edge_index,
        edge_attr=edge_attr,
        node_type=node_type,
        return_noise=True,
    )
    assert isinstance(result, torch.Tensor), (
        "Expected a plain tensor (no noise tuple) when use_noise_head=False"
    )
    assert result.shape == (num_edges, 3)


def test_no_noise_head_with_input_noise_no_crash():
    """Model with use_noise_head=False does not crash when input_noise is provided.

    Covers the case where the trainer passes input_noise (sigma/alpha) but the
    model has no noise head — the noise projection path should be a no-op because
    no noise injection flags are set.
    """
    config = _make_minimal_model_config(use_noise_head=False)
    device = torch.device("cpu")
    model = ModelFactory.create_model(config, device)

    em = EdgeFeatureManager(config.model)
    num_nodes, num_edges, batch_size = 8, 12, 2
    x = torch.zeros(num_nodes * batch_size, 1)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, em.num_edge_feats)
    node_type = torch.tensor([9] + [1] * (num_nodes - 1))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Pass non-None input_noise even though use_noise_head=False
    input_noise = torch.zeros(1, 2)

    result = model(
        x[:num_nodes],
        edge_index,
        edge_attr=edge_attr,
        node_type=node_type,
        batch=batch,
        input_noise=input_noise,
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape == (num_edges, 3)
