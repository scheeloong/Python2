#_author_: Timothy Ng (ECF username: ngtimoth)
# and Soon Chee Loong (ECF username: soonchee)

def get_distances(filename):
    """
    This function takes in a name of a file as a string and
    returns a dictionary where the keys of the dictionary
    are the original cities listed in the file,
    and the values are lists of tuples that
    represent all the neighbouring cities
    and it's respective distances from each original cities.
    """

    # Create an empty list to store the original cities and
    # their neighbouring cities with their distances in a list
    line_list = []

    # Create an empty string used to concatenate
    # the strings into the name of a city.
    s = ''

    # Create an empty list to store the
    # distances of the neighbouring cities.
    city2_distance_list = []

    # Create an empty list to store the
    # original cities of the list.
    city1_list = []

    # Create an empty dictionary to store the
    # original cities as keys and their neighbouring
    # cities with their distances as values.
    city_distance_dict = {}

    # Create a temporary list to store an
    # original city and its distances with every
    # possible neighbouring cities.
    temp_list = []

    # Open the file given by the user.
    filename = open(filename)

    #Create a loop to convert the data obtained from the file into a list.
    for line in filename:
        # Remove the unwanted whitespaces before and after a sentence
        # along with newlines, "\n" between the lines
        j = line.strip()

        # Separate j at it's colon and append it
        # into list, line_list.
        line_list.append(j.split(":"))

    filename.close()

    # Create a for loop which adds the original cities into city1_list.
    for i in range(len(line_list)):

        #add the name of original cities from the line_list into city1_list.
        city1_list.append(line_list[i][0])

        # a string that represents the neighbouring cities
        # and their distances from each original list
        j = line_list[i][1]

        for k in range(len(j)):
            # Creates a for loop which adds the neighbouring
            # cities and their distances into
            # city2_distance_list
            for l in range(len(j[k])):

                # If the single string m is a number string:
                if j[k][l] in "1234567890":

                    # Append the number string into this list
                    city2_distance_list.append(s.strip())
                    s = ''

                    # Convert the single digit string into an integer
                    city2_distance_list.append(int(j[k][l]))

                # Else, concatenating the single alphabet strings together
                # until a digit string appears,which means a string which
                # represents a city is formed. This will create a
                # dictionary with the given original cities
                # and its distances from its given neighbouring
                # cities from the file.
                else:
                    s += j[k][l]

    for i in range(len(city1_list)):
        #if the original city is already in the dictionary:
        if city1_list[i] in city_distance_dict.keys():

            g = city_distance_dict[city1_list[i]]

            # Append the distances of the new values of new
            # neighbouring cities to the original cities
            # already found in the dictionary, as a tuple.
            g.append(tuple(city2_distance_list[2 * i: 2 * i + 2]))

            # Creates a new definition for this existing key
            # to include the values extra neighbouring cities
            # that refer to this same key.
            city_distance_dict[city1_list[i]] = g

        # If the original city is not in the dictionary,
        # create the dictionary with the city name as
        # the key and its distances as the values.
        else:
            city_distance_dict[city1_list[i]] = \
            [tuple(city2_distance_list[2 * i: 2 * i + 2])]

    # Creates a for loop that updates the dictionary by giving the distances
    # between the original cities and all of its possible
    # neighbouring cities that is given indirectly from the file.
    for i in range(0, len(city2_distance_list), 2):
        # if the neighbouring city is already a
        # key in the dictionary:
        if city2_distance_list[i] in city_distance_dict.keys():

            g = city_distance_dict[city2_distance_list[i]]

            # Append the neighbouring cities into the
            # temporary list.
            temp_list.append((city1_list[i / 2]))

            # Append the distances from these cities
            # into the temporary list.
            temp_list.append(city2_distance_list[i + 1])

            # Convert the temporary list into a list of tuples
            # and append it to the variable g.
            g.append(tuple(temp_list))

            # Update the keys in the dictionary.
            city_distance_dict[city2_distance_list[i]] = g

            # Make the temporary list to be an empty
            # list again.
            temp_list = []

        #if the neighbouring cities is not a key in the dictionary:
        else:

            # Appends the neighbouring cities
            # from city1_list into the temporary list.
            temp_list.append((city1_list[i / 2]))

            # Appends the distances form these cities
            # from city2_distance_list
            # into the temporary list.
            temp_list.append(city2_distance_list[i + 1])

            #update the keys for the dictionary.
            city_distance_dict[city2_distance_list[i]] = [tuple(temp_list)]

            # Makes the temporary list to be an
            # empty list  again
            temp_list = []

    return city_distance_dict

def get_closest(unvisited):
    """
    This function returns the shortest distance of
    an unvisited city to its neighbouring city
    when it is given a list of tuples that provides
    the distances between an unvisited city
    and all of its neighbouring cities.
    """

    # Set the variable, min_length to be the
    # distance of its first
    # neighbouring city listed in the tuple.
    min_length = unvisited[0][1]

    tuple_index = 0

    for i in range(len(unvisited)):

        # If the new distance is smaller than
        # the current minimum distance,
        # substitute the new distance as the
        # current the minimum distance.
        if unvisited[i][1] < min_length:
            min_length = unvisited[i][1]
            tuple_index = i

    # Returns the smallest distance tuple from
    # the lists of tuples given.
    return unvisited[tuple_index]

def visit_next(visited, unvisited, distance):
    """
    This function takes in 2 lists of tuples,
    one for visited cities and their distance
    from the outbreak city, and another for the unvisited cities
    and it's distance from the outbreak city.
    it moves the city with the shortest distance, which is the distance
    from the visited city to its nearest neighbouring city,
    from the unvisited list to the visited list.
    It also updates the unvisited list with new paths
    that exist from this moved city. The function
    also takes in the dictionary of distances
    between the neighbouring cities,
    in order to add new paths and update
    existing paths in the unvisited list.
    """

    # Takes in the visited cities as a list
    visited_list = visited

    # Takes in the unvisited cities as a list
    unvisited_list = unvisited

    # Gets the shortest distance from
    # the unvisited list into the visited list
    shortest_distance = get_closest(unvisited_list)
    unvisited_list_loop = unvisited_list[:]
    for i in unvisited_list_loop:
        # Takes the shortest distance in the unvisited list
        if shortest_distance == i:
            # Adds the shortest distance from the unvisited list
            # to the visited list
            visited_list.append(i)
	    unvisited_list.remove(i) 
	    for j in unvisited_list:
		if shortest_distance[0] == j[0]:
            	    unvisited_list.remove(j) # remove all same cities from the unvisited file 
            # Gets the neighbouring cities of the shortest distance
            # if the neighbouring cities is already in the visited_list:
            new_distance = distance[shortest_distance[0]]
            #if the new neighbouring cities are already in
            #the visited lists, remove them
	    new_distance_loop = new_distance[:]
	    for k in new_distance_loop:
                for m in visited_list:
		    if k[0] == m[0]:
                        new_distance.remove(k)
                        break

            #this for loop adds to the total distance
            #of the new neighbouring cities from the
            #point of origin of the outbreak.
            for element in new_distance:
		# Append the new distance to that path as the shortest distance from 
		# the source town to the town that was just expanded
		# and add the distance from the just expanded town to this new town and append it to unvisited. 
                x = element[1]
                y = shortest_distance[1]
                m = x + y
                unvisited_list.append((element[0], m))
	    break # Only 1 city is needed for this 
    return visited_list, unvisited_list
