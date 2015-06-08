""" 
Test some of the functions in outbreak.py.
"""

# Read in the solution code.
import outbreak

def test_sample():
    """ A simple test of functions get_distances and visit_next.
    """
    print outbreak.visit_next([('Toronto',0)], [('New York',3),('Mexico City', 7), ('San Francisco',6)],outbreak.get_distances("cities.txt"))
if __name__ == "__main__":
    test_sample()
