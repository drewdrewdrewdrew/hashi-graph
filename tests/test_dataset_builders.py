from hashi_puzzle_solver.data import (
    NODE_TYPE_COMPONENT_META,
    NODE_TYPE_ISLAND_END,
    NODE_TYPE_ISLAND_START,
    HashiDataset,
)


def test_static_2n_node_structure() -> None:
    """Verify that HashiDataset always produces N islands + N component nodes."""
    # Create a dummy raw directory and a minimal puzzle file if needed,
    # but we can also mock graph_info or test builder methods directly.
    # For speed, let's test the builders on a mock object.

    # Mocking self for HashiDataset methods
    class MockDataset:
        use_capacity = True
        use_structural_degree = True
        use_unused_capacity = True
        use_conflict_status = False
        use_closeness_centrality = False
        use_articulation_points = False
        use_spectral_features = False
        use_structural_degree_nsew = False
        use_meta_node = True
        use_row_col_meta = False
        use_conflict_edges = False
        use_meta_mesh = False
        use_meta_row_col_edges = False
        use_edge_labels_as_features = False
        use_cut_edges = False
        use_potential_crossing = False
        use_component_meta = False
        use_continuous_edge_labels = False

        _get_feature_schema = HashiDataset._get_feature_schema
        _build_island_nodes = HashiDataset._build_island_nodes
        _build_meta_nodes = HashiDataset._build_meta_nodes
        _build_component_meta_nodes = HashiDataset._build_component_meta_nodes

    ds = MockDataset()
    schema = ds._get_feature_schema()

    graph_info = {
        "nodes": [
            {"id": 0, "pos": [0, 0], "n": 2},
            {"id": 1, "pos": [1, 0], "n": 2}
        ],
        "edges": []
    }

    # 1. Islands
    res = ds._build_island_nodes(graph_info, schema)
    x, node_type, pos_list, _id_to_idx, _bridges = res
    assert x.size(0) == 2
    assert (node_type >= NODE_TYPE_ISLAND_START).all()
    assert (node_type <= NODE_TYPE_ISLAND_END).all()

    # 2. Global Meta
    res_meta = ds._build_meta_nodes(
        graph_info, schema, x, node_type, pos_list,
    )
    x, node_type, pos_list, _meta_info = res_meta
    assert x.size(0) == 3  # 2 islands + 1 global

    # 3. Component Metas (should add N=2 nodes)
    res_comp = ds._build_component_meta_nodes(
        graph_info, schema, x, node_type, pos_list,
    )
    x, node_type, pos_list, comp_indices = res_comp

    assert x.size(0) == 5  # 2 islands + 1 global + 2 component metas
    assert (node_type == NODE_TYPE_COMPONENT_META).sum() == 2
    assert len(comp_indices) == 2
    assert pos_list[-1] == [-2000.0, -2000.0]  # Verify sentinel position
