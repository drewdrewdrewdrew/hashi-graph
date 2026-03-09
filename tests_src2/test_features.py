"""Tests for feature management logic."""

from hashi_puzzle_solver.models.config import ModelConfig
from hashi_puzzle_solver.models.features import EdgeFeatureManager, NodeFeatureManager


def test_node_feature_manager_schema():
    """Test that NodeFeatureManager generates the correct schema."""
    config = ModelConfig(
        use_capacity=True,
        use_structural_degree=True,
        use_unused_capacity=True,
        use_conflict_status=False,
        use_closeness_centrality=True,
        use_articulation_points=False,
        use_spectral_features=True,
    )
    manager = NodeFeatureManager(config)

    assert manager.num_node_feats == 7  # cap(1) + deg(1) + unused(1) + close(1) + spec(3)
    assert manager.get_idx("capacity") == 0
    assert manager.get_idx("structural_degree") == 1
    assert manager.get_idx("unused_capacity") == 2
    assert manager.get_idx("closeness_centrality") == 3
    assert manager.get_idx("spectral_1") == 4
    assert manager.get_idx("spectral_2") == 5
    assert manager.get_idx("spectral_3") == 6

    assert manager.has_feature("capacity") is True
    assert manager.has_feature("conflict_status") is False


def test_edge_feature_manager_schema():
    """Test that EdgeFeatureManager generates the correct schema."""
    config = ModelConfig(
        use_categorical_edge_types=True,
        use_edge_labels_as_features=True,
        use_cut_edges=True,
        use_potential_crossing=False,
        use_continuous_edge_labels=True,
    )
    manager = EdgeFeatureManager(config)

    # inv_dx(1), inv_dy(1) = 2
    # bridge_label(1), is_labeled(1) = 2
    # is_cut_edge(1) = 1
    # bridge_logits(3) = 3
    # Total = 8
    assert manager.num_edge_feats == 8

    assert manager.get_idx("inv_dx") == 0
    assert manager.get_idx("inv_dy") == 1
    assert manager.get_idx("bridge_label") == 2
    assert manager.get_idx("is_labeled") == 3
    assert manager.get_idx("is_cut_edge") == 4
    assert manager.get_idx("bridge_logits") == 5
