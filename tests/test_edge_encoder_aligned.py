"""Tests for Phase 4: EdgeEncoder with use_aligned_label_encoding flag.

Covers:
- output_dim calculation with flag off (no regression) and on
- forward output shape with flag on
- bridge_label / is_labeled embeddings produce distinct vectors
- gradient flow end-to-end
- error raised when use_aligned_label_encoding=True but use_edge_labels_as_features=False
- parity between output_dim property and actual tensor width
- various feature flag combinations
"""

import torch
import pytest

from hashi_puzzle_solver.models.config import ModelConfig
from hashi_puzzle_solver.models.encoders import EdgeEncoder
from hashi_puzzle_solver.models.features import EdgeFeatureManager


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_config(**overrides: object) -> ModelConfig:
    """Minimal ModelConfig with sensible dims for fast tests."""
    defaults = dict(
        use_categorical_edge_types=False,
        use_edge_labels_as_features=True,
        use_conflict_edges=False,
        use_meta_mesh=False,
        use_meta_row_col_edges=False,
        use_cut_edges=False,
        use_potential_crossing=False,
        use_continuous_edge_labels=False,
        use_component_meta=False,
        use_boundary_flag=False,
        # Embedding dims small for speed
        distance_embedding_dim=8,
        edge_type_embedding_dim=4,
        bridge_label_embedding_dim=8,
        is_labeled_embedding_dim=4,
        # Aligned label encoding is OFF by default
        use_aligned_label_encoding=False,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_edge_attr(fm: EdgeFeatureManager, num_edges: int = 8) -> torch.Tensor:
    """Synthetic edge_attr matching the feature manager schema."""
    attr = torch.rand(num_edges, fm.num_edge_feats)
    if fm.has_feature("bridge_label"):
        attr[:, fm.get_idx("bridge_label")] = torch.randint(0, 3, (num_edges,)).float()
    if fm.has_feature("is_labeled"):
        attr[:, fm.get_idx("is_labeled")] = torch.randint(0, 2, (num_edges,)).float()
    return attr


# ── regression: flag off ──────────────────────────────────────────────────────


def test_flag_off_output_dim_unchanged() -> None:
    """With use_aligned_label_encoding=False, output_dim matches legacy count.

    Layout (use_categorical_edge_types=False, use_edge_labels_as_features=True):
      distance_projector(8) + is_meta(1) + bridge_label(1) + is_labeled(1) = 11
    """
    cfg = _make_config(use_aligned_label_encoding=False)
    fm = EdgeFeatureManager(cfg)
    enc = EdgeEncoder(cfg, fm)

    # Enumerate all raw pass-through columns: everything except inv_dx, inv_dy, bridge_logits
    raw_count = sum(
        1 for name in fm.edge_map
        if name not in {"inv_dx", "inv_dy", "bridge_logits"}
    )
    expected = cfg.distance_embedding_dim + raw_count
    assert enc.output_dim == expected, (
        f"Expected output_dim={expected}, got {enc.output_dim}"
    )


def test_flag_off_forward_shape_matches_output_dim() -> None:
    """output_dim property equals actual tensor width when flag is off."""
    cfg = _make_config(use_aligned_label_encoding=False)
    fm = EdgeFeatureManager(cfg)
    enc = EdgeEncoder(cfg, fm)
    edge_attr = _make_edge_attr(fm, num_edges=6)

    out = enc(edge_attr)
    assert out.shape == (6, enc.output_dim)


# ── flag on: shape and output_dim ─────────────────────────────────────────────


def test_flag_on_output_dim_replaces_raw_with_embeddings() -> None:
    """When flag is on, bridge_label (8) + is_labeled (4) replace 2 raw cols.

    Layout (use_categorical_edge_types=False, use_edge_labels_as_features=True):
      Flag off:  distance(8) + is_meta(1) + bridge_label(1) + is_labeled(1) = 11
      Flag on:   distance(8) + is_meta(1) + bridge_label_emb(8) + is_labeled_emb(4) = 21
    """
    cfg_off = _make_config(use_aligned_label_encoding=False)
    cfg_on = _make_config(
        use_aligned_label_encoding=True,
        bridge_label_embedding_dim=8,
        is_labeled_embedding_dim=4,
    )
    fm_off = EdgeFeatureManager(cfg_off)
    fm_on = EdgeFeatureManager(cfg_on)
    enc_off = EdgeEncoder(cfg_off, fm_off)
    enc_on = EdgeEncoder(cfg_on, fm_on)

    # Non-label raw cols (everything except bridge_label/is_labeled/inv_dx/inv_dy/bridge_logits)
    non_label_raw = sum(
        1 for name in fm_off.edge_map
        if name not in {"inv_dx", "inv_dy", "bridge_logits", "bridge_label", "is_labeled"}
    )
    expected_off = cfg_off.distance_embedding_dim + non_label_raw + 2  # +2 for raw label cols
    expected_on = (
        cfg_on.distance_embedding_dim
        + non_label_raw
        + cfg_on.bridge_label_embedding_dim
        + cfg_on.is_labeled_embedding_dim
    )
    assert enc_off.output_dim == expected_off, f"flag-off: expected {expected_off}, got {enc_off.output_dim}"
    assert enc_on.output_dim == expected_on, f"flag-on: expected {expected_on}, got {enc_on.output_dim}"
    assert enc_on.output_dim > enc_off.output_dim


def test_flag_on_forward_output_shape() -> None:
    """Forward output shape equals output_dim property when flag is on."""
    cfg = _make_config(
        use_aligned_label_encoding=True,
        bridge_label_embedding_dim=8,
        is_labeled_embedding_dim=4,
    )
    fm = EdgeFeatureManager(cfg)
    enc = EdgeEncoder(cfg, fm)
    edge_attr = _make_edge_attr(fm, num_edges=10)

    out = enc(edge_attr)
    assert out.shape == (10, enc.output_dim), (
        f"Expected (10, {enc.output_dim}), got {out.shape}"
    )


def test_flag_on_output_dim_property_matches_tensor() -> None:
    """output_dim property always matches actual tensor width (flag on, various configs)."""
    combos = [
        dict(use_categorical_edge_types=False, use_conflict_edges=True),
        dict(use_categorical_edge_types=True),
        dict(use_cut_edges=True, use_potential_crossing=True),
        dict(use_continuous_edge_labels=True, logit_embedding_dim=8),
    ]
    for overrides in combos:
        cfg = _make_config(use_aligned_label_encoding=True, **overrides)
        fm = EdgeFeatureManager(cfg)
        enc = EdgeEncoder(cfg, fm)
        edge_attr = _make_edge_attr(fm, num_edges=5)
        out = enc(edge_attr)
        assert out.shape[1] == enc.output_dim, (
            f"Mismatch for config {overrides}: property={enc.output_dim}, "
            f"actual={out.shape[1]}"
        )


# ── embedding distinctness ────────────────────────────────────────────────────


def test_bridge_label_embeddings_are_distinct() -> None:
    """Embedding(3) for bridge_label produces distinct vectors for 0, 1, 2."""
    torch.manual_seed(0)
    cfg = _make_config(use_aligned_label_encoding=True, bridge_label_embedding_dim=8)
    fm = EdgeFeatureManager(cfg)
    enc = EdgeEncoder(cfg, fm)

    total = fm.num_edge_feats
    bl_idx = fm.get_idx("bridge_label")

    base = torch.zeros(3, total)
    base[:, bl_idx] = torch.tensor([0.0, 1.0, 2.0])

    out = enc(base)

    assert not torch.allclose(out[0], out[1]), "bridge_label 0 and 1 should differ"
    assert not torch.allclose(out[1], out[2]), "bridge_label 1 and 2 should differ"
    assert not torch.allclose(out[0], out[2]), "bridge_label 0 and 2 should differ"


def test_is_labeled_embeddings_are_distinct() -> None:
    """Embedding(2) for is_labeled produces distinct vectors for 0 and 1."""
    torch.manual_seed(1)
    cfg = _make_config(use_aligned_label_encoding=True, is_labeled_embedding_dim=4)
    fm = EdgeFeatureManager(cfg)
    enc = EdgeEncoder(cfg, fm)

    total = fm.num_edge_feats
    il_idx = fm.get_idx("is_labeled")

    base = torch.zeros(2, total)
    base[:, il_idx] = torch.tensor([0.0, 1.0])

    out = enc(base)
    assert not torch.allclose(out[0], out[1]), "is_labeled 0 and 1 should differ"


# ── gradient flow ─────────────────────────────────────────────────────────────


def test_gradient_flow_through_label_embeddings() -> None:
    """Gradients reach bridge_label_embedding and is_labeled_embedding."""
    torch.manual_seed(2)
    cfg = _make_config(
        use_aligned_label_encoding=True,
        bridge_label_embedding_dim=8,
        is_labeled_embedding_dim=4,
    )
    fm = EdgeFeatureManager(cfg)
    enc = EdgeEncoder(cfg, fm)

    edge_attr = _make_edge_attr(fm, num_edges=12)
    out = enc(edge_attr)
    out.sum().backward()

    for name, param in enc.named_parameters():
        assert param.grad is not None, f"No gradient for EdgeEncoder.{name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for EdgeEncoder.{name}"


def test_flag_off_no_gradient_to_label_embeddings() -> None:
    """With flag off, bridge_label_embedding and is_labeled_embedding don't exist."""
    cfg = _make_config(use_aligned_label_encoding=False)
    fm = EdgeFeatureManager(cfg)
    enc = EdgeEncoder(cfg, fm)

    assert not hasattr(enc, "bridge_label_embedding"), (
        "bridge_label_embedding should not exist when flag is off"
    )
    assert not hasattr(enc, "is_labeled_embedding"), (
        "is_labeled_embedding should not exist when flag is off"
    )


# ── error guard ───────────────────────────────────────────────────────────────


def test_flag_on_without_labels_raises() -> None:
    """use_aligned_label_encoding=True without use_edge_labels_as_features raises ValueError."""
    cfg = ModelConfig(
        use_aligned_label_encoding=True,
        use_edge_labels_as_features=False,
    )
    fm = EdgeFeatureManager(cfg)
    with pytest.raises(ValueError, match="use_edge_labels_as_features"):
        EdgeEncoder(cfg, fm)


# ── with categorical edge types ───────────────────────────────────────────────


def test_flag_on_with_categorical_edge_types() -> None:
    """Flag works correctly when categorical edge types are enabled (is_meta absent)."""
    cfg = _make_config(
        use_aligned_label_encoding=True,
        use_categorical_edge_types=True,
        edge_type_embedding_dim=4,
        bridge_label_embedding_dim=8,
        is_labeled_embedding_dim=4,
    )
    fm = EdgeFeatureManager(cfg)
    enc = EdgeEncoder(cfg, fm)
    edge_attr = _make_edge_attr(fm, num_edges=7)
    edge_type = torch.zeros(7, dtype=torch.long)

    out = enc(edge_attr, edge_type=edge_type)
    assert out.shape == (7, enc.output_dim)


def test_output_dim_parity_categorical_vs_non_categorical() -> None:
    """Aligned flag raises output_dim by the same amount regardless of categorical layout."""
    base_kwargs = dict(
        bridge_label_embedding_dim=8,
        is_labeled_embedding_dim=4,
        distance_embedding_dim=8,
    )

    cfg_cat_off = _make_config(use_categorical_edge_types=True, use_aligned_label_encoding=False, **base_kwargs)
    cfg_cat_on = _make_config(use_categorical_edge_types=True, use_aligned_label_encoding=True, **base_kwargs)
    cfg_non_off = _make_config(use_categorical_edge_types=False, use_aligned_label_encoding=False, **base_kwargs)
    cfg_non_on = _make_config(use_categorical_edge_types=False, use_aligned_label_encoding=True, **base_kwargs)

    def enc(cfg: ModelConfig) -> EdgeEncoder:
        return EdgeEncoder(cfg, EdgeFeatureManager(cfg))

    delta_cat = enc(cfg_cat_on).output_dim - enc(cfg_cat_off).output_dim
    delta_non = enc(cfg_non_on).output_dim - enc(cfg_non_off).output_dim

    expected_delta = 8 + 4 - 2  # embedding dims gained minus 2 raw cols lost
    assert delta_cat == expected_delta, f"Categorical delta={delta_cat}, expected {expected_delta}"
    assert delta_non == expected_delta, f"Non-categorical delta={delta_non}, expected {expected_delta}"
