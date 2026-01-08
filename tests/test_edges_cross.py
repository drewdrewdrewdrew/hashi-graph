"""Test edges_cross functionality."""


def edges_cross(
    edge1: tuple[tuple[float, float], tuple[float, float]],
    edge2: tuple[tuple[float, float], tuple[float, float]],
    _g: object = None,
) -> bool:
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
    is_ab_horizontal = ay == by and ax != bx
    is_cd_horizontal = cy == dy and cx != dx
    is_ab_vertical = ax == bx and ay != by
    is_cd_vertical = cx == dx and cy != dy

    # Only cross if one horizontal and one vertical
    if not (
        (is_ab_horizontal and is_cd_vertical) or (is_ab_vertical and is_cd_horizontal)
    ):
        return False

    # Check if they actually intersect
    if is_ab_horizontal and is_cd_vertical:
        # AB is horizontal, CD is vertical
        h_y = ay
        v_x = cx
        h_x_range = (min(ax, bx), max(ax, bx))
        v_y_range = (min(cy, dy), max(cy, dy))

        return v_y_range[0] < h_y < v_y_range[1] and h_x_range[0] < v_x < h_x_range[1]

    if is_ab_vertical and is_cd_horizontal:
        # AB is vertical, CD is horizontal
        v_x = ax
        h_y = cy
        v_y_range = (min(ay, by), max(ay, by))
        h_x_range = (min(cx, dx), max(cx, dx))

        return h_x_range[0] < v_x < h_x_range[1] and v_y_range[0] < h_y < v_y_range[1]

    return False


def test_horizontal_vertical_crossing() -> None:
    """Test horizontal and vertical edges that cross."""
    edge1 = ((0, 0), (2, 0))  # Horizontal: A to B
    edge2 = ((1, -1), (1, 1))  # Vertical: C to D

    assert edges_cross(edge1, edge2), "Should detect crossing!"


def test_two_horizontal_edges() -> None:
    """Test two horizontal edges (don't cross)."""
    edge1 = ((0, 0), (2, 0))  # Horizontal
    edge2 = ((0, 1), (2, 1))  # Horizontal (parallel)

    assert not edges_cross(edge1, edge2), "Should not detect crossing!"


def test_two_vertical_edges() -> None:
    """Test two vertical edges (don't cross)."""
    edge1 = ((0, 0), (0, 2))  # Vertical
    edge2 = ((1, 0), (1, 2))  # Vertical (parallel)

    assert not edges_cross(edge1, edge2), "Should not detect crossing!"


def test_vertical_too_short() -> None:
    """Test horizontal and vertical that don't cross (vertical too short)."""
    edge1 = ((0, 0), (3, 0))  # Horizontal from (0,0) to (3,0)
    edge2 = ((1, 1), (1, 2))  # Vertical from (1,1) to (1,2) - doesn't reach y=0

    assert not edges_cross(edge1, edge2), "Should not detect crossing!"


def test_horizontal_too_short() -> None:
    """Test horizontal and vertical that don't cross (horizontal too short)."""
    edge1 = ((0, 0), (0.5, 0))  # Horizontal from (0,0) to (0.5,0) - doesn't reach x=1
    edge2 = ((1, -1), (1, 1))  # Vertical at x=1

    assert not edges_cross(edge1, edge2), "Should not detect crossing!"


def test_crossing_at_intersection_point() -> None:
    """Test crossing at exact intersection point."""
    edge1 = ((0, 0), (2, 0))  # Horizontal through (1, 0)
    edge2 = ((1, -1), (1, 1))  # Vertical through (1, 0)

    assert edges_cross(edge1, edge2), "Should detect crossing at intersection!"


def test_reversed_edge_order() -> None:
    """Test reversed edge order (should still work)."""
    edge1 = ((2, 0), (0, 0))  # Horizontal reversed
    edge2 = ((1, 1), (1, -1))  # Vertical reversed

    assert edges_cross(edge1, edge2), "Should detect crossing regardless of order!"


def test_diagonal_edge() -> None:
    """Test diagonal edge (should not cross with horizontal/vertical)."""
    edge1 = ((0, 0), (2, 0))  # Horizontal
    edge2 = ((0, 1), (2, 2))  # Diagonal (not horizontal or vertical)

    assert not edges_cross(edge1, edge2), "Should not detect crossing with diagonal!"


def test_touch_at_endpoint() -> None:
    """Test edges that touch at endpoint (should not cross)."""
    edge1 = ((0, 0), (1, 0))  # Horizontal to (1,0)
    edge2 = ((1, 0), (1, 2))  # Vertical from (1,0) - shares endpoint

    assert not edges_cross(edge1, edge2), (
        "Should not detect crossing at shared endpoint!"
    )
