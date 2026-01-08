import unittest
from src.tune_space import expand_trial_config

class TestTuneSpace(unittest.TestCase):
    def test_expand_structural_degree(self):
        params = {"structural_degree_mode": "nsew"}
        expanded = expand_trial_config(params)
        self.assertFalse(expanded["use_structural_degree"])
        self.assertTrue(expanded["use_structural_degree_nsew"])

    def test_expand_row_col_meta(self):
        params = {"row_col_meta_mode": "with_mesh"}
        expanded = expand_trial_config(params)
        self.assertTrue(expanded["use_row_col_meta"])
        self.assertTrue(expanded["use_meta_mesh"])
        self.assertFalse(expanded["use_meta_row_col_edges"])

    def test_expand_verification_degradation(self):
        # Case: row_col_meta_mode is "none", but verification is "full"
        params = {
            "row_col_meta_mode": "none",
            "verification_mode": "full"
        }
        expanded = expand_trial_config(params)
        self.assertTrue(expanded["use_verification_head"])
        self.assertTrue(expanded["verifier_use_puzzle_nodes"])
        # Should be False due to degradation
        self.assertFalse(expanded["verifier_use_row_col_meta_nodes"])

    def test_expand_verification_full(self):
        # Case: row_col_meta_mode is "basic", verification is "full"
        params = {
            "row_col_meta_mode": "basic",
            "verification_mode": "full"
        }
        expanded = expand_trial_config(params)
        self.assertTrue(expanded["verifier_use_row_col_meta_nodes"])

if __name__ == '__main__':
    unittest.main()



