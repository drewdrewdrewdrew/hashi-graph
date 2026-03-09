import pytest

from hashi_puzzle_solver.data import FeatureSchema


def test_feature_schema_mapping() -> None:
    """Test that FeatureSchema correctly maps names to indices."""
    node_map = {"capacity": 0, "unused_capacity": 1}
    edge_map = {"inv_dx": 0, "inv_dy": 1, "is_meta": 2}
    schema = FeatureSchema(node_map, edge_map, 2, 3)

    assert schema.get_node_idx("capacity") == 0
    assert schema.get_node_idx("unused_capacity") == 1
    assert schema.get_edge_idx("inv_dx") == 0
    assert schema.get_edge_idx("is_meta") == 2
    assert schema.num_node_feats == 2
    assert schema.num_edge_feats == 3

    with pytest.raises(ValueError, match="Node feature 'invalid' not found"):
        schema.get_node_idx("invalid")

    with pytest.raises(ValueError, match="Edge feature 'invalid' not found"):
        schema.get_edge_idx("invalid")
