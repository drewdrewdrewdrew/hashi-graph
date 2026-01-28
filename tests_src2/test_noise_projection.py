import torch
import pytest
from hashi_puzzle_solver.models.config import HashiModelConfig
from hashi_puzzle_solver.models.factory import ModelFactory
from hashi_puzzle_solver.models.heads import EdgeHead
from hashi_puzzle_solver.models.features import NodeFeatureManager, EdgeFeatureManager

def test_noise_projection_shape():
    config = HashiModelConfig()
    config.model.noise_embedding_dim = 16
    device = torch.device("cpu")
    model = ModelFactory.create_model(config, device)
    
    input_noise = torch.randn(2, 2)  # 2 graphs, 2 noise features
    noise_emb = model.noise_projection(input_noise)
    assert noise_emb.shape == (2, 16)

def test_edge_head_with_noise():
    config = HashiModelConfig().model
    config.noise_embedding_dim = 16
    config.use_noise_in_prediction = True
    
    node_hidden_dim = 32
    edge_attr_dim = 8
    
    head = EdgeHead(config, node_hidden_dim, edge_attr_dim)
    
    # Create dummy data for 2 edges
    h = torch.randn(5, node_hidden_dim)
    edge_index = torch.tensor([[0, 2], [1, 3]])
    edge_attr = torch.randn(2, edge_attr_dim)
    noise_emb = torch.randn(1, 16) # Single graph
    node_type = torch.tensor([0, 0, 0, 0, 9]) # 9 is global meta
    batch = torch.zeros(5, dtype=torch.long)
    
    # Test forward
    out = head(h, edge_index, edge_attr=edge_attr, node_type=node_type, batch=batch, noise_emb=noise_emb)
    assert out.shape == (2, 3)

def test_full_model_noise_injection():
    config = HashiModelConfig()
    config.model.noise_embedding_dim = 16
    config.model.use_noise_in_message_passing = True
    config.model.use_noise_in_prediction = True
    config.model.use_noise_in_global_meta = True
    
    # Disable features that use Embedding to avoid index errors with randn
    config.model.use_capacity = False
    config.model.use_structural_degree = False
    config.model.use_conflict_status = False
    config.model.use_closeness_centrality = True
    config.model.use_articulation_points = True
    config.model.use_spectral_features = False
    
    device = torch.device("cpu")
    model = ModelFactory.create_model(config, device)
    
    nm = NodeFeatureManager(config.model)
    em = EdgeFeatureManager(config.model)
    
    x = torch.randn(5, nm.num_node_feats)
    edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
    edge_attr = torch.randn(3, em.num_edge_feats)
    batch = torch.tensor([0, 0, 0, 1, 1])
    node_type = torch.tensor([0, 1, 9, 0, 9]) # 9 is global meta
    input_noise = torch.randn(2, 2)
    
    # Forward pass
    out = model(
        x, 
        edge_index, 
        edge_attr=edge_attr, 
        batch=batch, 
        node_type=node_type, 
        input_noise=input_noise
    )
    
    assert out.shape == (3, 3) # 3 edges total, 3 classes

def test_no_noise_injection():
    config = HashiModelConfig()
    config.model.use_noise_in_message_passing = False
    config.model.use_noise_in_prediction = False
    config.model.use_noise_in_global_meta = False
    
    # Disable features that use Embedding
    config.model.use_capacity = False
    config.model.use_structural_degree = False
    config.model.use_conflict_status = False
    
    device = torch.device("cpu")
    model = ModelFactory.create_model(config, device)
    
    nm = NodeFeatureManager(config.model)
    em = EdgeFeatureManager(config.model)
    
    x = torch.randn(5, nm.num_node_feats)
    edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
    edge_attr = torch.randn(3, em.num_edge_feats)
    batch = torch.tensor([0, 0, 0, 1, 1])
    node_type = torch.tensor([0, 1, 9, 0, 9])
    
    # Should work without input_noise
    out = model(x, edge_index, edge_attr=edge_attr, batch=batch, node_type=node_type)
    assert out.shape == (3, 3)
