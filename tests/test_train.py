import unittest

import torch
from torch_geometric.data import Data

from hashi_puzzle_solver.masking import MaskingStrategy


class TestTrainLogic(unittest.TestCase):
    """Test training logic functions."""

    def test_get_masking_rate_schedules(self) -> None:
        """Test masking rate schedule calculation."""
        config = {
            "training": {
                "masking": {
                    "enabled": True,
                    "warmup_epochs": 10,
                    "cooldown_epochs": 10,
                    "start_rate": 0.0,
                    "end_rate": 1.0,
                    "schedule": "linear",
                }
            }
        }
        strategy = MaskingStrategy(config)
        total_epochs = 100

        # Warmup
        assert strategy.get_rate(5, total_epochs) == 0.0
        # Cooldown
        assert strategy.get_rate(95, total_epochs) == 1.0
        # Linear Midpoint: (50-10)/(100-10-10) = 40/80 = 0.5
        assert strategy.get_rate(50, total_epochs) == 0.5

    def test_apply_edge_label_masking(self) -> None:
        """Test edge label masking functionality."""
        # 2 nodes, 1 edge
        edge_attr = torch.tensor([[0.5, 0.0, 0.0, 1.0, 1.0]])
        x = torch.tensor([[5.0, 2.0, 5.0], [5.0, 2.0, 5.0]])  # [cap, struct, unused]
        edge_index = torch.tensor([[0], [1]])
        edge_mask = torch.tensor([True])

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, edge_mask=edge_mask,
        )
        config = {
            "model": {
                "use_capacity": True,
                "use_structural_degree": True,
                "use_unused_capacity": True,
                "use_edge_labels_as_features": True,
            },
        }
        strategy = MaskingStrategy(config)

        # Apply 100% masking
        masked_data = strategy.apply(data, 1.0, torch.device("cpu"))

        # Features 3 and 4 (label and is_labeled) should be 0
        assert masked_data.edge_attr[0, 3] == 0.0
        assert masked_data.edge_attr[0, 4] == 0.0
        # Unused capacity (idx 2) reset to 0, then incremented by masked label (1.0)
        assert masked_data.x[0, 2] == 1.0


if __name__ == "__main__":
    unittest.main()
