""" 
Test some of the functions in outbreak.py.
"""

# Read in the solution code.
import outbreak

def test_sample():
    """ A simple test of functions get_distances and visit_next.
    """
    # Start by visiting Toronto 
    unvisited = [('Toronto',0)]
    visited = []
    # Keep executing while the list is not empty 
    while unvisited:
    	visited,unvisited = outbreak.visit_next(visited,unvisited,outbreak.get_distances("cities.txt"))
    print visited
if __name__ == "__main__":
    test_sample()
