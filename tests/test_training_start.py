from collections.abc import Generator

import torch
from torch_geometric.data import Batch, Data

from hashi_puzzle_solver.engine import EpochMetrics, Trainer


def test_trainer_initialization_and_minimal_step() -> None:
    """Verify that Trainer can be initialized and run a single training step."""
    config = {
        "model": {
            "type": "gine",
            "node_embedding_dim": 8,
            "hidden_channels": 16,
            "num_layers": 2,
            "dropout": 0.1,
            "use_capacity": True,
            "use_structural_degree": True,
            "use_unused_capacity": True,
            "use_conflict_status": True,
            "use_global_meta_node": False,
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 1,
            "epochs": 1,
            "masking": {"enabled": False},
        }
    }
    device = torch.device("cpu")
    trainer = Trainer(config, device)

    # Manually setup model and optimizer since we won't use create_dataloader
    from hashi_puzzle_solver.models.factory import ModelFactory
    trainer.model = ModelFactory.create_model(config, device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.001)

    # Create dummy data
    # 2 nodes, 1 edge
    # Node features: capacity, struct_deg, unused_cap, conflict, closeness (5 features)
    x = torch.zeros((2, 5))
    x[:, 0] = torch.tensor([2, 2])  # Capacity
    x[:, 1] = torch.tensor([2, 2])  # Structural Degree
    x[:, 2] = torch.tensor([2, 2])  # Unused Capacity
    x[:, 3] = torch.tensor([0, 0])  # Conflict Status
    x[:, 4] = torch.tensor([0.5, 0.5])  # Closeness

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    y = torch.tensor([1, 1], dtype=torch.long)
    edge_mask = torch.tensor([True, True], dtype=torch.bool)

    # Node types (1-8 for islands)
    node_type = torch.tensor([2, 2], dtype=torch.long)

    data = Data(
        x=x, edge_index=edge_index, y=y, edge_mask=edge_mask,
        node_type=node_type, num_nodes=2
    )
    batch = Batch.from_data_list([data])

    # Mock a loader
    class MockLoader:
        def __init__(self, batch_data: Batch) -> None:
            self.batch_data = batch_data
            self.dataset = [batch_data]

        def __iter__(self) -> Generator[Batch, None, None]:
            yield self.batch_data

        def __len__(self) -> int:
            return 1

    loader = MockLoader(batch)

    metrics = trainer.run_epoch_one_shot(
        trainer.model,
        loader,
        training=True,
        optimizer=trainer.optimizer,
        masking_rate=0.0
    )

    assert isinstance(metrics, EpochMetrics)
    assert metrics.loss >= 0
    print("Minimal training step successful!")


if __name__ == "__main__":
    test_trainer_initialization_and_minimal_step()
