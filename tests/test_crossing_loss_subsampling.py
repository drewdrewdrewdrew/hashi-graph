import torch
from torch_geometric.data import Data, Batch
from hashi_puzzle_solver.utils import custom_collate_with_conflicts
from hashi_puzzle_solver.losses import compute_crossing_loss

def test_crossing_loss_preservation_in_subsampling():
    # 1. Create Data objects with conflicts
    # Graph 0: 2 edges, conflict between 0 and 1
    data0 = Data(
        x=torch.zeros((3, 1)),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_conflicts=[(0, 1)],
    )
    # Graph 1: 3 edges, NO conflict
    data1 = Data(
        x=torch.zeros((4, 1)),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        edge_conflicts=[],
    )
    # Graph 2: 2 edges, conflict between 0 and 1
    data2 = Data(
        x=torch.zeros((3, 1)),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_conflicts=[(0, 1)],
    )

    # 2. Collate into batch
    batch = custom_collate_with_conflicts([data0, data1, data2])
    
    # 3. Verify initial conflicts
    # Graph 0 edges: 0, 1 -> conflict (0, 1)
    # Graph 1 edges: 2, 3, 4 -> no conflict
    # Graph 2 edges: 5, 6 -> conflict (5, 6)
    assert hasattr(batch, "edge_conflict_index")
    assert batch.edge_conflict_index.size(1) == 2
    assert torch.equal(batch.edge_conflict_index, torch.tensor([[0, 5], [1, 6]], dtype=torch.long))

    # 4. Subsample the batch (e.g. take Graph 0 and Graph 2)
    indices = torch.tensor([0, 2], dtype=torch.long)
    # In PyG, slicing a Batch returns a list if we use [indices] 
    # and we need to from_data_list it back to a Batch for full functionality
    # but the diffusion_engine uses [indices] directly.
    
    # Let's see what happens if we use the same pattern as diffusion_engine
    subsampled_list = batch[indices]
    
    # When we from_data_list, we need to use custom_collate_with_conflicts 
    # to handle the edge_conflict_index.
    subsampled_batch = custom_collate_with_conflicts(subsampled_list)
    
    assert hasattr(subsampled_batch, "edge_conflict_index")
    assert subsampled_batch.edge_conflict_index.size(1) == 2
    assert torch.equal(subsampled_batch.edge_conflict_index, torch.tensor([[0, 2], [1, 3]], dtype=torch.long))
    
    # 6. Verify crossing loss is non-zero
    logits = torch.randn((4, 3)) # 4 edges total in subsampled batch
    # Set high probability for conflicting edges in Graph 0 (0 and 1)
    logits[0, 1] = 10.0
    logits[1, 1] = 10.0
    # Set high probability for conflicting edges in Graph 2 (2 and 3)
    logits[2, 1] = 10.0
    logits[3, 1] = 10.0
    
    loss = compute_crossing_loss(logits, subsampled_batch.edge_conflict_index, torch.ones(4, dtype=torch.bool))
    assert loss.item() > 0.0

if __name__ == "__main__":
    test_crossing_loss_preservation_in_subsampling()
