"""
Python wrapper for bridges_gen C executable.

Usage:
    from src.bridges_gen import generate_bridges
    
    # Generate a single easy 7x7 puzzle
    puzzles = generate_bridges(size=7, difficulty=0)
    
    # Generate 5 medium 10x10 puzzles
    puzzles = generate_bridges(count=5, size=10, difficulty=1)
    
    for puzzle in puzzles:
        print(f"Params: {puzzle.params}")
        print(f"Description: {puzzle.description}")
        print(f"Solution: {puzzle.solution}")
    
CLI Usage:
    python -m src.bridges_gen 7 0        # 7x7 puzzle, difficulty 0
    python -m src.bridges_gen 10 2      # 10x10 puzzle, difficulty 2
    python -m src.bridges_gen 7 0 --count 5  # Generate 5 puzzles
"""
import subprocess
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class BridgesPuzzle:
    """Represents a single Bridges puzzle."""
    params: str
    description: str
    solution: str
    
    def __repr__(self):
        return f"BridgesPuzzle(params={self.params}, desc_len={len(self.description)}, solution_len={len(self.solution)})"


def _get_bridges_gen_path() -> Path:
    """Get the path to the bridges_gen executable."""
    # Get the project root (assuming this file is in src/)
    project_root = Path(__file__).parent.parent
    bridges_gen_path = project_root / "generator" / "bridges_gen"
    
    if not bridges_gen_path.exists():
        raise FileNotFoundError(
            f"bridges_gen executable not found at {bridges_gen_path}. "
            "Please build it first with 'make bridges_gen' in the generator directory."
        )
    
    return bridges_gen_path


def generate_bridges(
    count: int = 1,
    width: Optional[int] = None,
    height: Optional[int] = None,
    size: Optional[int] = None,
    difficulty: int = 0,
    max_bridges: int = 2,
    islands_pct: Optional[int] = None,
    expansion: Optional[int] = None,
    allow_loops: bool = False,
    params: Optional[str] = None
) -> List[BridgesPuzzle]:
    """
    Generate Bridges puzzles using the C bridges_gen executable.
    
    Args:
        count: Number of puzzles to generate (default: 1)
        width: Puzzle width in cells (default: None, uses size if provided)
        height: Puzzle height in cells (default: None, uses size if provided)
        size: Puzzle size for square grids (creates size x size grid). 
              If provided, overrides width and height (default: None)
        difficulty: Difficulty level (default: 0)
            - 0: Easy - Only basic island-level logic required
            - 1: Medium - Requires connection-level logic (prevents loops, deduces required bridges)
            - 2: Hard - Requires group-level logic (prevents isolated subgraphs)
        max_bridges: Maximum number of bridges per connection (default: 2)
        islands_pct: Percentage of grid cells that should be islands (default: None, uses generator default)
        expansion: Expansion percentage for island placement (default: None, uses generator default)
        allow_loops: Whether to allow loops in the puzzle (default: False)
        params: Legacy parameter string for backward compatibility. If provided, 
                all other parameters are ignored (default: None)
    
    Returns:
        List of BridgesPuzzle objects containing params, description, and solution
    
    Examples:
        >>> # Generate a single easy 7x7 puzzle
        >>> puzzles = generate_bridges(size=7, difficulty=0)
        
        >>> # Generate 5 medium 10x10 puzzles
        >>> puzzles = generate_bridges(count=5, size=10, difficulty=1)
        
        >>> # Generate a hard rectangular puzzle
        >>> puzzles = generate_bridges(width=15, height=10, difficulty=2)
        
        >>> # Generate with custom island density
        >>> puzzles = generate_bridges(size=8, difficulty=1, islands_pct=25)
    """
    bridges_gen_path = _get_bridges_gen_path()
    
    # Build command
    cmd = [str(bridges_gen_path), str(count)]
    
    # Handle legacy params string for backward compatibility
    if params:
        cmd.append(params)
    else:
        # Build params string from individual arguments
        # Determine dimensions
        if size is not None:
            dimensions = f"{size}x{size}"
        elif width is not None and height is not None:
            dimensions = f"{width}x{height}"
        elif width is not None:
            dimensions = f"{width}x{width}"  # Default to square if only width provided
        elif height is not None:
            dimensions = f"{height}x{height}"  # Default to square if only height provided
        else:
            dimensions = None  # Use generator defaults
        
        if dimensions:
            # Build parameter string: {dim}[i{islands_pct}][e{expansion}]m{max_bridges}d{difficulty}
            params_parts = [dimensions]
            
            if islands_pct is not None:
                params_parts.append(f"i{islands_pct}")
            
            if expansion is not None:
                params_parts.append(f"e{expansion}")
            
            # max_bridges: m2 is standard, m3+ allows more bridges (typically for loops)
            if allow_loops and max_bridges < 3:
                max_bridges = 3  # Loops typically require at least m3
            params_parts.append(f"m{max_bridges}")
            
            params_parts.append(f"d{difficulty}")
            
            params_str = "".join(params_parts)
            cmd.append(params_str)
    
    # Run the executable
    # Clean environment to avoid VIRTUAL_ENV conflicts
    env = os.environ.copy()
    env.pop('VIRTUAL_ENV', None)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=bridges_gen_path.parent,
            env=env
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"bridges_gen failed with exit code {e.returncode}.\n"
            f"stderr: {e.stderr}"
        ) from e
    
    # Parse output
    puzzles = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        
        # Format is: PARAMS:DESC,SOLUTION
        if ':' not in line:
            continue
        
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        
        params_str = parts[0]
        desc_and_solution = parts[1]
        
        # Split description and solution
        if ',' in desc_and_solution:
            desc, solution = desc_and_solution.split(',', 1)
        else:
            desc = desc_and_solution
            solution = ""
        
        puzzles.append(BridgesPuzzle(
            params=params_str,
            description=desc,
            solution=solution
        ))
    
    return puzzles


def generate_bridges_dict(
    count: int = 1,
    width: Optional[int] = None,
    height: Optional[int] = None,
    size: Optional[int] = None,
    difficulty: int = 0,
    max_bridges: int = 2,
    islands_pct: Optional[int] = None,
    expansion: Optional[int] = None,
    allow_loops: bool = False,
    params: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Generate Bridges puzzles and return as dictionaries.
    
    Same as generate_bridges() but returns dictionaries instead of BridgesPuzzle objects.
    See generate_bridges() for parameter documentation.
    
    Returns:
        List of dictionaries with keys: 'params', 'description', 'solution'
    """
    puzzles = generate_bridges(
        count=count,
        width=width,
        height=height,
        size=size,
        difficulty=difficulty,
        max_bridges=max_bridges,
        islands_pct=islands_pct,
        expansion=expansion,
        allow_loops=allow_loops,
        params=params
    )
    return [
        {
            'params': p.params,
            'description': p.description,
            'solution': p.solution
        }
        for p in puzzles
    ]


def build_params_string(dimensions: str, difficulty: int, allow_loops: bool = False) -> str:
    """
    Build a parameter string for bridges_gen.
    
    Args:
        dimensions: Puzzle dimensions, e.g., "7x7" or "10x10"
        difficulty: Difficulty level (0=easy, 1=medium, 2=hard, etc.)
        allow_loops: Whether to allow loops (default: False, i.e., no loops)
    
    Returns:
        Parameter string for bridges_gen, e.g., "7x7m2d0"
    """
    # Format: {dim}m2d{difficulty}
    # m2 appears to be a standard parameter (possibly max bridges per connection)
    # For no loops, we use m2 (which seems to be the default based on examples)
    # If loops were allowed, it might be m3 or a different parameter
    params = f"{dimensions}m2d{difficulty}"
    return params


def main():
    """CLI entry point for bridges_gen."""
    parser = argparse.ArgumentParser(
        description="Generate Bridges puzzles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 7 0              # Generate 1 easy 7x7 puzzle (no loops)
  %(prog)s 10 2             # Generate 1 hard 10x10 puzzle (no loops)
  %(prog)s 7 0 --count 5    # Generate 5 easy 7x7 puzzles
        """
    )
    
    parser.add_argument(
        "size",
        type=int,
        help="Puzzle size (creates a square grid, e.g., 7 for 7x7, 10 for 10x10)"
    )
    
    parser.add_argument(
        "difficulty",
        type=int,
        help="Difficulty level (0=easy, 1=medium, 2=hard, etc.)"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of puzzles to generate (default: 1)"
    )
    
    parser.add_argument(
        "--allow-loops",
        action="store_true",
        help="Allow loops in puzzles (default: False, i.e., no loops)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["full", "params", "description", "solution"],
        default="full",
        help="Output format (default: full)"
    )
    
    args = parser.parse_args()
    
    # Validate size
    if args.size <= 0:
        parser.error(f"Size must be a positive integer, got: {args.size}")
    
    # Generate puzzles using new API
    try:
        puzzles = generate_bridges(
            count=args.count,
            size=args.size,
            difficulty=args.difficulty,
            allow_loops=args.allow_loops
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate puzzles via bridges_gen: {exc}"
        ) from exc
    
    # Output results
    for puzzle in puzzles:
        if args.output_format == "full":
            print(f"{puzzle.params}:{puzzle.description},{puzzle.solution}")
        elif args.output_format == "params":
            print(puzzle.params)
        elif args.output_format == "description":
            print(puzzle.description)
        elif args.output_format == "solution":
            print(puzzle.solution)


if __name__ == "__main__":
    main()

