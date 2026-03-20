import torch
import pytest
from torch_geometric.data import Data, Batch
from hashi_puzzle_solver.utils.ar_utils import detect_components, rewire_hierarchical_edges

def test_detect_components_probabilistic():
    """Test detect_components with probabilistic AM logic."""
    num_islands = 2
    # 0-1 puzzle edge
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    current_bridges = torch.zeros(2)
    node_type = torch.tensor([1, 1], dtype=torch.long)
    
    # Logits: high confidence for bridge 1 (index 1)
    # Probs will be softmax of [0, 10, 0] -> [~0, ~1, ~0]
    logits = torch.tensor([
        [0.0, 10.0, 0.0],
        [0.0, 10.0, 0.0]
    ])
    
    # 1. High margin, high confidence -> Should merge
    margin = 0.5
    reps = detect_components(num_islands, edge_index, current_bridges, node_type, logits=logits, margin=margin)
    assert reps[0] == reps[1]
    
    # 2. High margin, low confidence -> Should NOT merge
    # Logits: [0, 1, 0.9] -> Probs: [0.1, 0.46, 0.44] -> Margin: 0.46 - 0.44 = 0.02
    logits_low_conf = torch.tensor([
        [0.0, 1.0, 0.9],
        [0.0, 1.0, 0.9]
    ])
    reps_low = detect_components(num_islands, edge_index, current_bridges, node_type, logits=logits_low_conf, margin=margin)
    assert reps_low[0] != reps_low[1]
    
    # 3. Argmax is 0 -> Should NOT merge even if high confidence
    logits_zero = torch.tensor([
        [10.0, 0.0, 0.0],
        [10.0, 0.0, 0.0]
    ])
    reps_zero = detect_components(num_islands, edge_index, current_bridges, node_type, logits=logits_zero, margin=margin)
    assert reps_zero[0] != reps_zero[1]

def test_rewire_hierarchical_probabilistic():
    """Test rewire_hierarchical_edges with probabilistic AM/BM logic."""
    # 2 islands, 2 comp metas, 1 global meta
    node_type = torch.tensor([1, 1, 11, 11, 9], dtype=torch.long)
    x = torch.zeros((5, 10))
    
    # Edges: 
    # 0: 0-1 (puzzle)
    # 1: 1-0 (puzzle)
    # 2: 0-2 (meta)
    # 3: 2-0 (meta)
    # 4: 1-3 (meta)
    # 5: 3-1 (meta)
    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 3],
        [1, 0, 2, 0, 3, 1]
    ], dtype=torch.long)
    
    edge_attr = torch.zeros((6, 3))
    edge_attr[2:, 2] = 1.0 # is_meta
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_type=node_type)
    data.ptr = torch.tensor([0, 5])
    data.batch = torch.zeros(5, dtype=torch.long)
    data.num_graphs = 1
    
    batch = Batch.from_data_list([data])
    
    model_config = {
        "use_component_meta": True,
        "use_hierarchical_component_meta": True,
        "component_merge_margin": 0.5
    }
    
    # Scenario: Low confidence edge -> BM logic
    # This should result in islands being in different components, 
    # and a comp-comp edge being created.
    logits_bm = torch.zeros((6, 3))
    # Puzzle edges 0, 1: [0, 1, 0.9] -> Low margin
    logits_bm[0:2] = torch.tensor([0.0, 1.0, 0.9])
    
    updated_batch = rewire_hierarchical_edges(batch, model_config, current_bridges=torch.zeros(6), logits=logits_bm)
    
    # Check that islands remain in separate components (0 connects to 2, 1 connects to 3)
    fwd_mask = (updated_batch.node_type[updated_batch.edge_index[0]] <= 8) & (updated_batch.node_type[updated_batch.edge_index[1]] == 11)
    meta_dests = updated_batch.edge_index[1, fwd_mask]
    assert 2 in meta_dests and 3 in meta_dests
    
    # Check for comp-comp edges (2-3 or 3-2)
    new_edges = updated_batch.edge_index[:, 6:]
    found_comp_comp = False
    for j in range(new_edges.size(1)):
        src, dst = new_edges[0, j].item(), new_edges[1, j].item()
        if (src == 2 and dst == 3) or (src == 3 and dst == 2):
            found_comp_comp = True
            break
    assert found_comp_comp

    # Scenario: High confidence edge -> AM logic
    # This should merge components. No comp-comp edges should exist between them.
    logits_am = torch.zeros((6, 3))
    logits_am[0:2] = torch.tensor([0.0, 10.0, 0.0])
    
    # Need a fresh batch because rewire_hierarchical_edges might modify in-place
    batch_am = Batch.from_data_list([data.clone()])
    updated_batch_am = rewire_hierarchical_edges(batch_am, model_config, current_bridges=torch.zeros(6), logits=logits_am)
    
    # Check that islands are merged (both connect to 2)
    fwd_mask_am = (updated_batch_am.node_type[updated_batch_am.edge_index[0]] <= 8) & (updated_batch_am.node_type[updated_batch_am.edge_index[1]] == 11)
    assert torch.all(updated_batch_am.edge_index[1, fwd_mask_am] == 2)
    
    # No comp-comp edges between 2 and 3 should exist (only 2-4, 4-2)
    # Total edges = 6 (standard) + 2 (comp-global 2-4, 4-2) = 8
    assert updated_batch_am.edge_index.size(1) == 6 + 2
