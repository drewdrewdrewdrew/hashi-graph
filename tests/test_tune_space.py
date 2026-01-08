import unittest

from hashi_puzzle_solver.tune_space import expand_trial_config


class TestTuneSpace(unittest.TestCase):
    """Unit tests for tune space configuration expansion."""

    def test_expand_structural_degree(self) -> None:
        """Test expansion of structural degree configuration."""
        params = {"structural_degree_mode": "nsew"}
        expanded = expand_trial_config(params)
        assert not expanded["use_structural_degree"]
        assert expanded["use_structural_degree_nsew"]

    def test_expand_row_col_meta(self) -> None:
        """Test expansion of row/col meta configuration."""
        params = {"row_col_meta_mode": "with_mesh"}
        expanded = expand_trial_config(params)
        assert expanded["use_row_col_meta"]
        assert expanded["use_meta_mesh"]
        assert not expanded["use_meta_row_col_edges"]

    def test_expand_verification_degradation(self) -> None:
        """Test verification configuration with degradation."""
        # Case: row_col_meta_mode is "none", but verification is "full"
        params = {
            "row_col_meta_mode": "none",
            "verification_mode": "full",
        }
        expanded = expand_trial_config(params)
        assert expanded["use_verification_head"]
        assert expanded["verifier_use_puzzle_nodes"]
        # Should be False due to degradation
        assert not expanded["verifier_use_row_col_meta_nodes"]

    def test_expand_verification_full(self) -> None:
        """Test verification configuration in full mode."""
        # Case: row_col_meta_mode is "basic", verification is "full"
        params = {
            "row_col_meta_mode": "basic",
            "verification_mode": "full",
        }
        expanded = expand_trial_config(params)
        assert expanded["verifier_use_row_col_meta_nodes"]


if __name__ == "__main__":
    unittest.main()
