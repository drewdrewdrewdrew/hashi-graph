"""
Hierarchical constraint expansion for Optuna tuning.

Converts mode enums into flat boolean configuration flags.
"""

from typing import Any


def expand_structural_degree_mode(mode: str) -> dict[str, bool]:
    """Expand structural degree mode into discrete flags."""
    return {
        "none": {"use_structural_degree": False, "use_structural_degree_nsew": False},
        "count": {"use_structural_degree": True, "use_structural_degree_nsew": False},
        "nsew": {"use_structural_degree": False, "use_structural_degree_nsew": True},
    }[mode]


def expand_row_col_meta_mode(mode: str) -> dict[str, bool]:
    """Expand row/col meta mode into discrete flags."""
    return {
        "none": {
            "use_row_col_meta": False,
            "use_meta_mesh": False,
            "use_meta_row_col_edges": False,
        },
        "basic": {
            "use_row_col_meta": True,
            "use_meta_mesh": False,
            "use_meta_row_col_edges": False,
        },
        "with_mesh": {
            "use_row_col_meta": True,
            "use_meta_mesh": True,
            "use_meta_row_col_edges": False,
        },
        "with_cross": {
            "use_row_col_meta": True,
            "use_meta_mesh": False,
            "use_meta_row_col_edges": True,
        },
        "full": {
            "use_row_col_meta": True,
            "use_meta_mesh": True,
            "use_meta_row_col_edges": True,
        },
    }[mode]


def expand_verification_mode(mode: str, row_col_mode: str) -> dict[str, bool]:
    """Expand verification mode with graceful degradation if row/col nodes absent."""
    effective_mode = mode
    if mode == "full" and row_col_mode == "none":
        # Degradation: can't use row/col meta in verifier if they don't exist
        effective_mode = "meta_plus_puzzle"

    return {
        "disabled": {
            "use_verification_head": False,
            "verifier_use_puzzle_nodes": False,
            "verifier_use_row_col_meta_nodes": False,
        },
        "meta_only": {
            "use_verification_head": True,
            "verifier_use_puzzle_nodes": False,
            "verifier_use_row_col_meta_nodes": False,
        },
        "meta_plus_puzzle": {
            "use_verification_head": True,
            "verifier_use_puzzle_nodes": True,
            "verifier_use_row_col_meta_nodes": False,
        },
        "full": {
            "use_verification_head": True,
            "verifier_use_puzzle_nodes": True,
            "verifier_use_row_col_meta_nodes": True,
        },
    }[effective_mode]


def expand_trial_config(trial_params: dict[str, Any]) -> dict[str, Any]:
    """Expand all mode enums in a trial's parameter dictionary."""
    expanded = trial_params.copy()

    if "structural_degree_mode" in trial_params:
        mode = expanded.pop("structural_degree_mode")
        expanded.update(expand_structural_degree_mode(mode))

    row_col_mode = trial_params.get("row_col_meta_mode", "none")
    if "row_col_meta_mode" in trial_params:
        expanded.pop("row_col_meta_mode")
        expanded.update(expand_row_col_meta_mode(row_col_mode))

    if "verification_mode" in trial_params:
        mode = expanded.pop("verification_mode")
        expanded.update(expand_verification_mode(mode, row_col_mode))

    return expanded
