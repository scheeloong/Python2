""" 
Test some of the functions in outbreak.py.
"""

# Read in the solution code.
import outbreak

def test_sample():
    """ A simple test of functions get_distances and visit_next.
    """
    # Create the distance dictionary-- assumes the data file exists.
    distances = outbreak.get_distances("cities.txt")
    # Start off with the outbreak city on the unvisited list.
    visited = []
    unvisited = [("Toronto", 0)]
    # Transfer cities to the visited list until there are no cities left.
    while unvisited:
        outbreak.visit_next(visited, unvisited, distances)
    print visited
if __name__ == "__main__":
    test_sample()
