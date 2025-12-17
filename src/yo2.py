from src.bridges_gen import generate_bridges
from src.bridges_utils import convert_to_hashi_format
from hashi import print_puzzle, print_solution

# Generate a puzzle
puzzles = generate_bridges(count=1, size=5, difficulty=0)
puzzle = puzzles[0]

# Convert to hashi format
hashi_puzzle = convert_to_hashi_format(puzzle)

# Now you can use it with the hashi package
print_solution(hashi_puzzle)


"""
next steps:
- graph representation of the puzzle and solutions! look at your notebook, get help
- something to keep an eye on, do we need any kind of structural info as long as we set up the edges correclty?
- apparently graphormer isnt set up for edge classificaiont
"""