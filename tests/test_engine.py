import unittest

import torch

from hashi_puzzle_solver.models.factory import ModelFactory


class TestEngine(unittest.TestCase):
    """Test cases for the Trainer engine functionality."""

    def test_compute_edge_dim(self) -> None:
        """Test computation of edge dimension based on configuration."""
        config = {
            "model": {
                "use_conflict_edges": True,
                "use_meta_mesh": True,
                "use_meta_row_col_edges": True,
                "use_edge_labels_as_features": True,
                "use_cut_edges": False,
            },
        }
        # 3 (base) + 1 (conflict) + 1 (mesh) + 1 (row_col) + 2 (labels) = 8
        assert ModelFactory.compute_edge_dim(config) == 8

    def test_create_model_transformer(self) -> None:
        """Test creation of transformer model with verification head."""
        config = {
            "model": {
                "type": "transformer",
                "node_embedding_dim": 16,
                "hidden_channels": 32,
                "num_layers": 2,
                "use_verification_head": True,
                "use_global_meta_node": True,
                "use_capacity": True,
                "use_structural_degree": True,
                "use_unused_capacity": True,
                "use_conflict_status": True,
                "use_closeness_centrality": False,
                "use_row_col_meta": False,
            },
        }
        device = torch.device("cpu")
        model = ModelFactory.create_model(config, device)
        assert model.use_verification_head
        assert hasattr(model, "verify_mlp")


if __name__ == "__main__":
    unittest.main()
