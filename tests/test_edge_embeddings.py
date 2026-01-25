import torch
from hashi_puzzle_solver.models.factory import ModelFactory
from hashi_puzzle_solver.data import HashiDataset, FeatureSchema
import pytest

def test_edge_embeddings_shapes():
    """Verify that edge embeddings result in correct internal shapes."""
    node_embedding_dim = 32
    hidden_channels = 64
    
    config = {
        "model": {
            "type": "transformer",
            "node_embedding_dim": node_embedding_dim,
            "hidden_channels": hidden_channels,
            "num_layers": 2,
            "use_categorical_edge_types": True,
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_conflict_status": True,
            "use_global_meta_node": True,
            "use_edge_features_in_prediction": True,
            "use_distance": True,
            "use_conflict_edges": True,
            "use_continuous_edge_labels": True,
            "use_cut_edges": True,
            "use_potential_crossing": True
        }
    }
    
    device = torch.device("cpu")
    model = ModelFactory.create_model(config, device)
    
    # 1. Check compute_edge_dim
    edge_dim = ModelFactory.compute_edge_dim(config)
    # Expected: node_embedding_dim (32) + continuous features
    # Continuous: inv_dx(1), inv_dy(1), bridge_logits(3), cut_edge(1), potential_crossing(1)
    # Total continuous = 7.
    # Total edge_dim = 32 + 7 = 39.
    assert edge_dim == node_embedding_dim + 7
    assert model.edge_dim == edge_dim

    # 2. Generate dummy data
    num_nodes = 10
    num_edges = 20
    
    # Node features for NodeEncoder:
    # 0: capacity (0-15)
    # 1: degree (0-15)
    # 2: unused capacity (continuous)
    # 3: conflict status (0-1)
    # 4: closeness centrality (continuous)
    # 5: articulation point (continuous)
    # 6-8: spectral features (continuous)
    x = torch.zeros((num_nodes, 9))
    x[:, 0] = torch.randint(0, 16, (num_nodes,)).float()
    x[:, 1] = torch.randint(0, 16, (num_nodes,)).float()
    x[:, 2] = torch.randn(num_nodes)
    x[:, 3] = torch.randint(0, 2, (num_nodes,)).float()
    x[:, 4] = torch.randn(num_nodes)
    x[:, 5] = torch.randn(num_nodes)
    x[:, 6:9] = torch.randn((num_nodes, 3))
    
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn((num_edges, 7)) # only continuous features
    edge_type = torch.randint(0, 9, (num_edges,))
    node_type = torch.randint(0, 12, (num_nodes,))
    
    # 3. Forward pass
    logits = model(x, edge_index, edge_attr=edge_attr, edge_type=edge_type, node_type=node_type)
    
    assert logits.shape == (num_edges, 3)
    
    # 4. Check backward compatibility (use_categorical_edge_types=False)
    config_old = {
        "model": config["model"].copy()
    }
    config_old["model"]["use_categorical_edge_types"] = False
    model_old = ModelFactory.create_model(config_old, device)
    
    edge_dim_old = ModelFactory.compute_edge_dim(config_old)
    # base(3: inv_dx, inv_dy, is_meta) + conflict(1) + labels_as_features(0) 
    # + cut(1) + potential(1) + continuous(3) = 9
    assert edge_dim_old == 9
    
    edge_attr_old = torch.randn((num_edges, 9))
    logits_old = model_old(x, edge_index, edge_attr=edge_attr_old, node_type=node_type)
    assert logits_old.shape == (num_edges, 3)

def test_dataset_edge_types():
    """Verify that HashiDataset generates edge_type when enabled."""
    from hashi_puzzle_solver.data import HashiDataset
    
    # Mock some minimal attributes for HashiDataset
    class MockDataset(HashiDataset):
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    ds = MockDataset(
        use_categorical_edge_types=True,
        use_capacity=True,
        use_structural_degree=True,
        use_structural_degree_nsew=False,
        use_unused_capacity=True,
        use_conflict_status=True,
        use_closeness_centrality=False,
        use_articulation_points=False,
        use_spectral_features=False,
        use_distance=True,
        use_edge_labels_as_features=False,
        use_conflict_edges=True,
        use_meta_mesh=False,
        use_meta_row_col_edges=False,
        use_cut_edges=False,
        use_potential_crossing=False,
        use_continuous_edge_labels=False,
        use_component_meta=False
    )
    
    schema = ds._get_feature_schema()
    
    # Continuous features should be at 0, 1 (inv_dx, inv_dy)
    assert schema.get_edge_idx("inv_dx") == 0
    assert schema.get_edge_idx("inv_dy") == 1
    
    # is_meta should NOT be in the schema when using categorical types
    with pytest.raises(ValueError):
        schema.get_edge_idx("is_meta")

    # Now check without categorical
    ds.use_categorical_edge_types = False
    schema_old = ds._get_feature_schema()
    assert schema_old.get_edge_idx("inv_dx") == 0
    assert schema_old.get_edge_idx("inv_dy") == 1
    assert schema_old.get_edge_idx("is_meta") == 2
