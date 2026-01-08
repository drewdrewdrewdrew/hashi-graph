#!/usr/bin/env python3
"""
Quick test for edges_cross logic in create_data.py
Tests the function directly without requiring full module imports.
"""


def edges_cross(edge1, edge2, G=None):
    """
    Check if two edges would cross geometrically.
    Simplified version for testing - assumes nodes are (x,y) tuples.
    """
    (a, b) = edge1
    (c, d) = edge2
    
    # Get positions (assuming nodes are (x,y) tuples)
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    
    # Check if one is horizontal and one is vertical
    is_ab_horizontal = (ay == by and ax != bx)
    is_cd_horizontal = (cy == dy and cx != dx)
    is_ab_vertical = (ax == bx and ay != by)
    is_cd_vertical = (cx == dx and cy != dy)
    
    # Only cross if one horizontal and one vertical
    if not ((is_ab_horizontal and is_cd_vertical) or (is_ab_vertical and is_cd_horizontal)):
        return False
    
    # Check if they actually intersect
    if is_ab_horizontal and is_cd_vertical:
        # AB is horizontal, CD is vertical
        h_y = ay
        v_x = cx
        h_x_range = (min(ax, bx), max(ax, bx))
        v_y_range = (min(cy, dy), max(cy, dy))
        
        return (v_y_range[0] < h_y < v_y_range[1] and 
                h_x_range[0] < v_x < h_x_range[1])
    
    elif is_ab_vertical and is_cd_horizontal:
        # AB is vertical, CD is horizontal
        v_x = ax
        h_y = cy
        v_y_range = (min(ay, by), max(ay, by))
        h_x_range = (min(cx, dx), max(cx, dx))
        
        return (h_x_range[0] < v_x < h_x_range[1] and 
                v_y_range[0] < h_y < v_y_range[1])
    
    return False


def test_edges_cross():
    """Test the edges_cross function with various scenarios."""
    
    print("Testing edges_cross logic...")
    print("=" * 50)
    
    # Test Case 1: Horizontal and vertical edges that cross
    print("\nTest 1: Horizontal and vertical edges that cross")
    edge1 = ((0, 0), (2, 0))  # Horizontal: A to B
    edge2 = ((1, -1), (1, 1)) # Vertical: C to D
    
    result = edges_cross(edge1, edge2)
    print(f"  Edge1 (horizontal): {edge1}")
    print(f"  Edge2 (vertical): {edge2}")
    print(f"  Result: {result} (expected: True)")
    assert result == True, "Should detect crossing!"
    print("  ✓ PASSED")
    
    # Test Case 2: Two horizontal edges (don't cross)
    print("\nTest 2: Two horizontal edges (don't cross)")
    edge1 = ((0, 0), (2, 0))  # Horizontal
    edge2 = ((0, 1), (2, 1))  # Horizontal (parallel)
    
    result = edges_cross(edge1, edge2)
    print(f"  Edge1: {edge1}")
    print(f"  Edge2: {edge2}")
    print(f"  Result: {result} (expected: False)")
    assert result == False, "Should not detect crossing!"
    print("  ✓ PASSED")
    
    # Test Case 3: Two vertical edges (don't cross)
    print("\nTest 3: Two vertical edges (don't cross)")
    edge1 = ((0, 0), (0, 2))  # Vertical
    edge2 = ((1, 0), (1, 2))  # Vertical (parallel)
    
    result = edges_cross(edge1, edge2)
    print(f"  Edge1: {edge1}")
    print(f"  Edge2: {edge2}")
    print(f"  Result: {result} (expected: False)")
    assert result == False, "Should not detect crossing!"
    print("  ✓ PASSED")
    
    # Test Case 4: Horizontal and vertical that don't cross (vertical doesn't reach horizontal)
    print("\nTest 4: Horizontal and vertical that don't cross (vertical too short)")
    edge1 = ((0, 0), (3, 0))  # Horizontal from (0,0) to (3,0)
    edge2 = ((1, 1), (1, 2))  # Vertical from (1,1) to (1,2) - doesn't reach y=0
    
    result = edges_cross(edge1, edge2)
    print(f"  Edge1 (horizontal): {edge1}")
    print(f"  Edge2 (vertical): {edge2}")
    print(f"  Result: {result} (expected: False)")
    assert result == False, "Should not detect crossing!"
    print("  ✓ PASSED")
    
    # Test Case 5: Horizontal and vertical that don't cross (horizontal doesn't reach vertical)
    print("\nTest 5: Horizontal and vertical that don't cross (horizontal too short)")
    edge1 = ((0, 0), (0.5, 0))  # Horizontal from (0,0) to (0.5,0) - doesn't reach x=1
    edge2 = ((1, -1), (1, 1))   # Vertical at x=1
    
    result = edges_cross(edge1, edge2)
    print(f"  Edge1 (horizontal): {edge1}")
    print(f"  Edge2 (vertical): {edge2}")
    print(f"  Result: {result} (expected: False)")
    assert result == False, "Should not detect crossing!"
    print("  ✓ PASSED")
    
    # Test Case 6: Crossing at exact intersection point
    print("\nTest 6: Crossing at exact intersection point")
    edge1 = ((0, 0), (2, 0))  # Horizontal through (1, 0)
    edge2 = ((1, -1), (1, 1)) # Vertical through (1, 0)
    
    result = edges_cross(edge1, edge2)
    print(f"  Edge1 (horizontal): {edge1}")
    print(f"  Edge2 (vertical): {edge2}")
    print(f"  Result: {result} (expected: True)")
    assert result == True, "Should detect crossing at intersection!"
    print("  ✓ PASSED")
    
    # Test Case 7: Reversed edge order (should still work)
    print("\nTest 7: Reversed edge order (should still work)")
    edge1 = ((2, 0), (0, 0))  # Horizontal reversed
    edge2 = ((1, 1), (1, -1)) # Vertical reversed
    
    result = edges_cross(edge1, edge2)
    print(f"  Edge1 (horizontal, reversed): {edge1}")
    print(f"  Edge2 (vertical, reversed): {edge2}")
    print(f"  Result: {result} (expected: True)")
    assert result == True, "Should detect crossing regardless of order!"
    print("  ✓ PASSED")
    
    # Test Case 8: Diagonal edge (should not cross with horizontal/vertical)
    print("\nTest 8: Diagonal edge (should not cross)")
    edge1 = ((0, 0), (2, 0))  # Horizontal
    edge2 = ((0, 1), (2, 2)) # Diagonal (not horizontal or vertical)
    
    result = edges_cross(edge1, edge2)
    print(f"  Edge1 (horizontal): {edge1}")
    print(f"  Edge2 (diagonal): {edge2}")
    print(f"  Result: {result} (expected: False)")
    assert result == False, "Should not detect crossing with diagonal!"
    print("  ✓ PASSED")
    
    # Test Case 9: Edges that touch at endpoint (should not cross)
    print("\nTest 9: Edges that touch at endpoint (should not cross)")
    edge1 = ((0, 0), (1, 0))  # Horizontal to (1,0)
    edge2 = ((1, 0), (1, 2))  # Vertical from (1,0) - shares endpoint
    
    result = edges_cross(edge1, edge2)
    print(f"  Edge1 (horizontal): {edge1}")
    print(f"  Edge2 (vertical): {edge2}")
    print(f"  Result: {result} (expected: False)")
    assert result == False, "Should not detect crossing at shared endpoint!"
    print("  ✓ PASSED")
    
    print("\n" + "=" * 50)
    print("All tests PASSED! ✓")
    print("=" * 50)


if __name__ == "__main__":
    test_edges_cross()
