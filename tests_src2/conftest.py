"""Ensure src2 package takes precedence over the installed src package."""

import sys
from pathlib import Path

# Insert src2 at position 0 so it shadows the installed src/hashi_puzzle_solver
src2_path = str(Path(__file__).parent.parent / "src2")
if src2_path not in sys.path:
    sys.path.insert(0, src2_path)
