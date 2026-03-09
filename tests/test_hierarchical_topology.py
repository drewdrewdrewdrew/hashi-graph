import torch
import pytest
from torch_geometric.data import Data, Batch
from hashi_puzzle_solver.ar_utils import rewire_hierarchical_edges

def test_rewire_hierarchical_edges_basic():
    """Test that hierarchical edges are created correctly in a simple scenario."""
    # 2 islands, comp metas, then global meta
    # Indices: 0,1 (islands), 2,3 (comp metas), 4 (global meta)
    node_type = torch.tensor([1, 1, 11, 11, 9], dtype=torch.long)
    x = torch.zeros((5, 10))
    
    # Edges: 
    # 0-1 (puzzle edge)
    # 0-2 (island 0 to comp meta 2)
    # 1-3 (island 1 to comp meta 3)
    edge_index = torch.tensor([
        [0, 1, 0, 2, 2, 0, 1, 3, 3, 1],
        [1, 0, 2, 0, 0, 2, 3, 1, 1, 3]
    ], dtype=torch.long)
    
    edge_attr = torch.zeros((edge_index.size(1), 3)) # inv_dx, inv_dy, is_meta
    edge_attr[2:, 2] = 1.0 # meta edges
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_type=node_type)
    data.ptr = torch.tensor([0, 5])
    data.batch = torch.zeros(5, dtype=torch.long)
    data.num_graphs = 1
    
    batch = Batch.from_data_list([data])
    
    # 1. No bridges -> 2 components {0}, {1}
    current_bridges = torch.zeros(edge_index.size(1))
    
    model_config = {
        "use_component_meta": True,
        "use_hierarchical_component_meta": True
    }
    
    # Before hierarchical: edges connect island to THEIR comp meta (0-2, 1-3)
    # After hierarchical:
    # - Comp 0 (node 2) <-> Comp 1 (node 3) because of boundary edge 0-1
    # - Comp 0 (node 2) <-> Global (node 4)
    # - Comp 1 (node 3) <-> Global (node 4)
    
    updated_batch = rewire_hierarchical_edges(batch, model_config, current_bridges=current_bridges)
    
    # Check new edges
    # Standard edges (10) + new edges
    # Boundary edge 0-1 (label 0) connects comp 0 and comp 1.
    # New comp-comp edges: 2-3, 3-2
    # New comp-global edges: 2-4, 4-2, 3-4, 4-3
    assert updated_batch.edge_index.size(1) == 10 + 2 + 4
    
    # Check that comp-comp edges are present
    new_edges = updated_batch.edge_index[:, 10:]
    found_comp_comp = False
    for j in range(new_edges.size(1)):
        if (new_edges[0, j] == 2 and new_edges[1, j] == 3) or (new_edges[0, j] == 3 and new_edges[1, j] == 2):
            found_comp_comp = True
            break
    assert found_comp_comp
    
    # Check that comp-global edges are present
    found_comp_global = 0
    for j in range(new_edges.size(1)):
        src, dst = new_edges[0, j].item(), new_edges[1, j].item()
        if (src == 2 and dst == 4) or (src == 4 and dst == 2) or (src == 3 and dst == 4) or (src == 4 and dst == 3):
            found_comp_global += 1
    assert found_comp_global == 4

def test_rewire_hierarchical_edges_connected():
    """Test that when islands are connected, they share a component meta and no boundary edges exist."""
    # Indices: 0,1 (islands), 2,3 (comp metas), 4 (global meta)
    node_type = torch.tensor([1, 1, 11, 11, 9], dtype=torch.long)
    x = torch.zeros((5, 10))
    
    # 0-1 (puzzle edge)
    # 0-2 (island 0 to comp meta 2)
    # 1-3 (island 1 to comp meta 3)
    edge_index = torch.tensor([
        [0, 1, 0, 2, 2, 0, 1, 3, 3, 1],
        [1, 0, 2, 0, 0, 2, 3, 1, 1, 3]
    ], dtype=torch.long)
    
    edge_attr = torch.zeros((edge_index.size(1), 3))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_type=node_type)
    data.ptr = torch.tensor([0, 5])
    data.batch = torch.zeros(5, dtype=torch.long)
    data.num_graphs = 1
    
    batch = Batch.from_data_list([data])
    
    # 1 bridge between 0 and 1 -> 1 component {0, 1} with rep 0
    current_bridges = torch.zeros(edge_index.size(1))
    current_bridges[0] = 1 # 0-1 bridge
    current_bridges[1] = 1 # 1-0 bridge
    
    model_config = {
        "use_component_meta": True,
        "use_hierarchical_component_meta": True
    }
    
    updated_batch = rewire_hierarchical_edges(batch, model_config, current_bridges=current_bridges)
    
    # Standard edges (10) + new edges
    # Both islands now connect to comp meta 2 (rep 0). Comp meta 3 is inactive.
    # No boundary edges (0-1 is NOT boundary because bridges > 0).
    # New comp-global edges: 2-4, 4-2 (only 1 active rep)
    assert updated_batch.edge_index.size(1) == 10 + 2
    
    # Check that both islands connect to node 2
    fwd_mask = (updated_batch.node_type[updated_batch.edge_index[0]] <= 8) & (updated_batch.node_type[updated_batch.edge_index[1]] == 11)
    assert torch.all(updated_batch.edge_index[1, fwd_mask] == 2)
    
    # New edges: 2-4 and 4-2
    new_edges = updated_batch.edge_index[:, 10:]
    assert ((new_edges[0, 0] == 2 and new_edges[1, 0] == 4) and (new_edges[0, 1] == 4 and new_edges[1, 1] == 2)) or \
           ((new_edges[0, 0] == 4 and new_edges[1, 0] == 2) and (new_edges[0, 1] == 2 and new_edges[1, 1] == 4))

def test_rewire_hierarchical_backwards_compatibility():
    """Test that it behaves like rewire_component_meta_edges_batch if flag is False."""
    # Indices: 0,1 (islands), 2,3 (comp metas), 4 (global meta)
    node_type = torch.tensor([1, 1, 11, 11, 9], dtype=torch.long)
    x = torch.zeros((5, 10))
    edge_index = torch.tensor([
        [0, 1, 0, 2, 2, 0, 1, 3, 3, 1],
        [1, 0, 2, 0, 0, 2, 3, 1, 1, 3]
    ], dtype=torch.long)
    edge_attr = torch.zeros((edge_index.size(1), 3))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_type=node_type)
    data.ptr = torch.tensor([0, 5])
    data.batch = torch.zeros(5, dtype=torch.long)
    data.num_graphs = 1
    
    batch = Batch.from_data_list([data])
    current_bridges = torch.zeros(edge_index.size(1))
    
    model_config = {
        "use_component_meta": True,
        "use_hierarchical_component_meta": False
    }
    
    updated_batch = rewire_hierarchical_edges(batch, model_config, current_bridges=current_bridges)
    
    # No new edges should be added
    assert updated_batch.edge_index.size(1) == 10
