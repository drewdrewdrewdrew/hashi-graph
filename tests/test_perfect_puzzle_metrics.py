"""
Unit tests for perfect puzzle accuracy calculations.
"""
import torch
from src.train_utils import calculate_perfect_puzzle_accuracy, calculate_batch_perfect_puzzles


def test_perfect_puzzle_accuracy_all_correct():
    """Test when all puzzles are perfectly solved."""
    # 2 puzzles, each with 3 edges
    predictions = torch.tensor([0, 1, 2, 0, 1, 2])
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    edge_masks = torch.ones(6, dtype=torch.bool)
    batch_indices = torch.tensor([0, 0, 0, 1, 1, 1])
    
    acc, perfect, total = calculate_perfect_puzzle_accuracy(
        predictions, targets, edge_masks, batch_indices
    )
    
    assert acc == 1.0, f"Expected 1.0, got {acc}"
    assert perfect == 2, f"Expected 2 perfect puzzles, got {perfect}"
    assert total == 2, f"Expected 2 total puzzles, got {total}"
    print("✓ Test passed: all correct")


def test_perfect_puzzle_accuracy_none_correct():
    """Test when no puzzles are perfectly solved."""
    # 2 puzzles, each with 3 edges, all wrong
    predictions = torch.tensor([1, 2, 0, 1, 2, 0])
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    edge_masks = torch.ones(6, dtype=torch.bool)
    batch_indices = torch.tensor([0, 0, 0, 1, 1, 1])
    
    acc, perfect, total = calculate_perfect_puzzle_accuracy(
        predictions, targets, edge_masks, batch_indices
    )
    
    assert acc == 0.0, f"Expected 0.0, got {acc}"
    assert perfect == 0, f"Expected 0 perfect puzzles, got {perfect}"
    assert total == 2, f"Expected 2 total puzzles, got {total}"
    print("✓ Test passed: none correct")


def test_perfect_puzzle_accuracy_partial():
    """Test when some puzzles are perfect and some are not."""
    # Puzzle 0: all correct (3/3)
    # Puzzle 1: 2/3 correct (not perfect)
    predictions = torch.tensor([0, 1, 2, 0, 1, 0])  # Last edge wrong in puzzle 1
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    edge_masks = torch.ones(6, dtype=torch.bool)
    batch_indices = torch.tensor([0, 0, 0, 1, 1, 1])
    
    acc, perfect, total = calculate_perfect_puzzle_accuracy(
        predictions, targets, edge_masks, batch_indices
    )
    
    assert acc == 0.5, f"Expected 0.5, got {acc}"
    assert perfect == 1, f"Expected 1 perfect puzzle, got {perfect}"
    assert total == 2, f"Expected 2 total puzzles, got {total}"
    print("✓ Test passed: partial correct")


def test_perfect_puzzle_accuracy_with_masking():
    """Test that edge masks are properly respected."""
    # 2 puzzles, each with 3 original edges + 1 meta edge
    predictions = torch.tensor([0, 1, 2, 0, 0, 1, 2, 0])
    targets = torch.tensor([0, 1, 2, 0, 0, 1, 2, 0])
    # Only original edges should be counted
    edge_masks = torch.tensor([True, True, True, False, True, True, True, False])
    batch_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    acc, perfect, total = calculate_perfect_puzzle_accuracy(
        predictions, targets, edge_masks, batch_indices
    )
    
    assert acc == 1.0, f"Expected 1.0, got {acc}"
    assert perfect == 2, f"Expected 2 perfect puzzles, got {perfect}"
    assert total == 2, f"Expected 2 total puzzles, got {total}"
    print("✓ Test passed: with masking")


def test_calculate_batch_perfect_puzzles():
    """Test the batch wrapper function with logits."""
    # Create logits for 2 puzzles, 3 edges each
    # Puzzle 0: all correct
    # Puzzle 1: one wrong
    logits = torch.tensor([
        [10.0, -5.0, -5.0],  # Class 0
        [-5.0, 10.0, -5.0],  # Class 1
        [-5.0, -5.0, 10.0],  # Class 2
        [10.0, -5.0, -5.0],  # Class 0
        [10.0, -5.0, -5.0],  # Class 0 (wrong, should be 1)
        [-5.0, -5.0, 10.0],  # Class 2
    ])
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    edge_masks = torch.ones(6, dtype=torch.bool)
    batch_indices = torch.tensor([0, 0, 0, 1, 1, 1])
    
    acc, perfect, total = calculate_batch_perfect_puzzles(
        logits, targets, edge_masks, batch_indices
    )
    
    assert acc == 0.5, f"Expected 0.5, got {acc}"
    assert perfect == 1, f"Expected 1 perfect puzzle, got {perfect}"
    assert total == 2, f"Expected 2 total puzzles, got {total}"
    print("✓ Test passed: batch wrapper")


def test_empty_puzzle():
    """Test edge case with empty input."""
    predictions = torch.tensor([])
    targets = torch.tensor([])
    edge_masks = torch.tensor([], dtype=torch.bool)
    batch_indices = torch.tensor([])
    
    acc, perfect, total = calculate_perfect_puzzle_accuracy(
        predictions, targets, edge_masks, batch_indices
    )
    
    assert acc == 0.0, f"Expected 0.0, got {acc}"
    assert perfect == 0, f"Expected 0 perfect puzzles, got {perfect}"
    assert total == 0, f"Expected 0 total puzzles, got {total}"
    print("✓ Test passed: empty puzzle")


def run_all_tests():
    """Run all tests."""
    print("Running perfect puzzle accuracy tests...\n")
    
    test_perfect_puzzle_accuracy_all_correct()
    test_perfect_puzzle_accuracy_none_correct()
    test_perfect_puzzle_accuracy_partial()
    test_perfect_puzzle_accuracy_with_masking()
    test_calculate_batch_perfect_puzzles()
    test_empty_puzzle()
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    run_all_tests()




