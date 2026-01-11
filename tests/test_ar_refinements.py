import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from hashi_puzzle_solver.ar_engine import ARTrainer, ARState
from hashi_puzzle_solver.ar_utils import get_edge_feature_indices

def test_get_edge_feature_indices():
    model_config = {
        "use_conflict_edges": True,
        "use_edge_labels_as_features": True,
        "use_cut_edges": True
    }
    indices = get_edge_feature_indices(model_config)
    assert indices["inv_dx"] == 0
    assert indices["inv_dy"] == 1
    assert indices["is_meta"] == 2
    assert indices["is_conflict"] == 3
    assert indices["bridge_label"] == 4
    assert indices["is_labeled"] == 5
    assert indices["is_cut_edge"] == 6

def test_ar_trainer_edge_feature_update():
    # Setup mock data
    num_nodes = 4
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    num_edges = edge_index.size(1)
    
    # 2 islands, 2 meta edges (one-way for simplicity in test)
    node_type = torch.tensor([1, 2, 9, 10], dtype=torch.long)
    edge_mask = torch.tensor([True, True, False, False], dtype=torch.bool)
    
    # Initial edge features (3 base + 2 label features)
    edge_attr = torch.zeros((num_edges, 5), dtype=torch.float)
    
    data = Data(
        x=torch.zeros((num_nodes, 10)),
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=node_type,
        edge_mask=edge_mask,
        y=torch.tensor([1, 1, 0, 0], dtype=torch.long)
    )
    
    config = {
        "model": {
            "type": "transformer",
            "node_embedding_dim": 64,
            "hidden_channels": 128,
            "num_layers": 2,
            "use_edge_labels_as_features": True,
            "use_unused_capacity": True,
            "use_capacity": True,
            "use_structural_degree": True,
            "use_conflict_status": True
        },
        "training": {
            "ar_k": 2,
            "ar_steps": 1,
            "learning_rate": 0.001,
            "batch_size": 1
        }
    }
    
    device = torch.device("cpu")
    
    # Mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(1))
        def forward(self, x, edge_index, edge_attr=None, batch=None, node_type=None):
            # Return dummy logits [num_edges, 2]
            return torch.zeros((edge_index.size(1), 2), device=x.device)
    
    model = MockModel()
    trainer = ARTrainer(model, config, device)
    
    # Verify index mapping
    assert trainer.bridge_label_idx == 3
    assert trainer.is_labeled_idx == 4
    
    # Setup ARState manually
    puzzle_state = ARState(data, device)
    puzzle_state.current_bridges = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    
    # Collate into Batch
    collated_data = Batch.from_data_list([puzzle_state.data])
    current_bridges_batch = puzzle_state.current_bridges
    
    # Manually run the update logic we added to run_epoch
    row, col = collated_data.edge_index
    is_puzzle_edge = (
        (collated_data.node_type[row] <= 8)
        & (collated_data.node_type[row] > 0)
        & (collated_data.node_type[col] <= 8)
        & (collated_data.node_type[col] > 0)
    )
    
    collated_data.edge_attr[is_puzzle_edge, trainer.bridge_label_idx] = (
        current_bridges_batch[is_puzzle_edge].float()
    )
    collated_data.edge_attr[is_puzzle_edge, trainer.is_labeled_idx] = 1.0
    
    # Verify
    # Edges 0 and 1 are puzzle edges. current_bridges for them are 1.
    assert collated_data.edge_attr[0, 3] == 1.0
    assert collated_data.edge_attr[1, 3] == 1.0
    assert collated_data.edge_attr[0, 4] == 1.0
    assert collated_data.edge_attr[1, 4] == 1.0
    # Edges 2 and 3 are NOT puzzle edges (connect to meta nodes 9 and 10)
    assert collated_data.edge_attr[2, 3] == 0.0
    assert collated_data.edge_attr[3, 3] == 0.0

if __name__ == "__main__":
    test_get_edge_feature_indices()
    test_ar_trainer_edge_feature_update()
    print("Tests passed!")
