"""
Utilities for converting bridges_gen puzzle format to hashi package format.

The bridges_gen format uses:
- Description: Run-length encoded string (e.g., "3a2c3b4f1c3c4")
- Solution: String with format ";Lx1,y1,x2,y2,count;Mx,y;..." 
- Params: String like "7x7i30e10m2d0"

The hashi package format uses:
- Dictionary with 'width', 'height', 'difficulty', 'islands', 'solution'
- islands: List[{'x': int, 'y': int, 'count': int}]
- solution: Tuple[(x1, y1, x2, y2, '-' or '='), ...]
"""
import re
from typing import Dict, List, Tuple, Any, Optional
from .bridges_gen import BridgesPuzzle


def parse_params(params_str: str) -> Dict[str, int]:
    """
    Parse bridges_gen params string (e.g., "7x7i30e10m2d0").
    
    Returns:
        Dictionary with 'width', 'height', 'difficulty', etc.
    """
    result = {}
    
    # Parse width x height
    match = re.match(r'(\d+)x(\d+)', params_str)
    if match:
        result['width'] = int(match.group(1))
        result['height'] = int(match.group(2))
        params_str = params_str[match.end():]
    
    # Parse optional parameters
    if 'i' in params_str:
        match = re.search(r'i(\d+)', params_str)
        if match:
            result['islands_pct'] = int(match.group(1))
    
    if 'e' in params_str:
        match = re.search(r'e(\d+)', params_str)
        if match:
            result['expansion'] = int(match.group(1))
    
    if 'm' in params_str:
        match = re.search(r'm(\d+)', params_str)
        if match:
            result['max_bridges'] = int(match.group(1))
    
    if 'd' in params_str:
        match = re.search(r'd(\d+)', params_str)
        if match:
            result['difficulty'] = int(match.group(1))
    
    return result


def parse_description(desc: str, width: int, height: int) -> List[Dict[str, int]]:
    """
    Parse bridges_gen description string to extract islands.
    
    The description is run-length encoded:
    - Digits '0'-'9' = island counts 0-9
    - Letters 'A'-'G' = island counts 10-16
    - Letters 'a'-'z' = runs of empty spaces (a=1, b=2, ..., z=26)
    - Grid is scanned row by row, left to right
    
    Args:
        desc: Description string (e.g., "3a2c3b4f1c3c4")
        width: Puzzle width
        height: Puzzle height
    
    Returns:
        List of islands as {'x': int, 'y': int, 'count': int}
    """
    islands = []
    pos = 0  # Position in the grid (0-indexed, row-major)
    
    i = 0
    while i < len(desc):
        char = desc[i]
        
        if char.isdigit():
            # Island with count 0-9
            count = int(char)
            x = pos % width
            y = pos // width
            islands.append({'x': x, 'y': y, 'count': count})
            pos += 1
            i += 1
        elif 'A' <= char <= 'G':
            # Island with count 10-16
            count = 10 + (ord(char) - ord('A'))
            x = pos % width
            y = pos // width
            islands.append({'x': x, 'y': y, 'count': count})
            pos += 1
            i += 1
        elif 'a' <= char <= 'z':
            # Run of empty spaces
            run_length = ord(char) - ord('a') + 1
            pos += run_length
            i += 1
        else:
            # Unknown character, skip
            i += 1
    
    return islands


def parse_solution(solution_str: str) -> List[Tuple[int, int, int, int, str]]:
    """
    Parse bridges_gen solution string to extract bridges.
    
    Format: ";Lx1,y1,x2,y2,count;Mx,y;..."
    - L = Line/bridge: (x1, y1) to (x2, y2) with count bridges
    - M = Mark (can be ignored for solution)
    - count: 1 = single bridge ('-'), 2 = double bridge ('=')
    
    Args:
        solution_str: Solution string (e.g., ";L0,3,4,3,2;L0,3,0,6,2;M0,3;...")
    
    Returns:
        List of bridges as (x1, y1, x2, y2, '-' or '=') tuples
    """
    bridges = []
    
    if not solution_str:
        return bridges
    
    # Split by semicolon and process each command
    parts = solution_str.split(';')
    
    for part in parts:
        if not part:
            continue
        
        if part.startswith('L'):
            # Line/bridge: Lx1,y1,x2,y2,count
            match = re.match(r'L(\d+),(\d+),(\d+),(\d+),(\d+)', part)
            if match:
                x1 = int(match.group(1))
                y1 = int(match.group(2))
                x2 = int(match.group(3))
                y2 = int(match.group(4))
                count = int(match.group(5))
                
                # Convert count to bridge type: 1 = '-', 2 = '='
                bridge_type = '=' if count >= 2 else '-'
                bridges.append((x1, y1, x2, y2, bridge_type))
        # M commands (marks) are ignored for solution
    
    return bridges


def convert_to_hashi_format(puzzle: BridgesPuzzle) -> Dict[str, Any]:
    """
    Convert a BridgesPuzzle to the hashi package format.
    
    Args:
        puzzle: BridgesPuzzle object from bridges_gen
    
    Returns:
        Dictionary in hashi package format:
        {
            'width': int,
            'height': int,
            'difficulty': int,
            'islands': List[{'x': int, 'y': int, 'count': int}],
            'solution': Tuple[(x1, y1, x2, y2, '-' or '='), ...]
        }
    """
    # Parse params to get dimensions and difficulty
    params = parse_params(puzzle.params)
    width = params.get('width', 7)  # Default to 7 if not found
    height = params.get('height', 7)
    difficulty = params.get('difficulty', 0)
    
    # Parse description to get islands
    islands = parse_description(puzzle.description, width, height)
    
    # Parse solution to get bridges
    bridges = parse_solution(puzzle.solution)
    
    return {
        'width': width,
        'height': height,
        'difficulty': difficulty,
        'islands': islands,
        'solution': tuple(bridges)
    }


def convert_bridges_to_hashi(
    puzzles: List[BridgesPuzzle]
) -> List[Dict[str, Any]]:
    """
    Convert a list of BridgesPuzzle objects to hashi format.
    
    Args:
        puzzles: List of BridgesPuzzle objects
    
    Returns:
        List of dictionaries in hashi package format
    """
    return [convert_to_hashi_format(puzzle) for puzzle in puzzles]

