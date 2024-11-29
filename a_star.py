import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, dist


class Node:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position # (x,y)

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

# This function return the path of the search


def return_path(current_node, maze):
    path = []
    no_rows, no_columns = np.shape(maze)
    # here we create the initialized result maze with -1 in every position
    result = [[-1 for i in range(no_columns)] for j in range(no_rows)]
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    # Return reversed path as we need to show from start to end path
    path = path[::-1]
    start_value = 0
    # we update the path of start to end found by A-star serch with every step incremented by 1
    for i in range(len(path)):
        result[path[i][0]][path[i][1]] = start_value
        start_value += 1

    return path

def manhattan_dist(p, q):
    assert len(p) == len(q)
    assert len(p) == 2
    return sum([abs(p_i - q_i) for p_i, q_i in zip(p, q)])

def search(maze, start, end):

    print("searching ....")

    maze = maze.T

    """
        Returns a list of tuples as a path from the given start to the given end in the given maze
        :param maze:
        :param cost
        :param start:
        :param end:
        :return:
    """
    use_manhattan = False # uses Manhattan distance if true otherwise use Euclidean

    # TODO PART 4 Create start and end node with initized values for g, h and f
    # Use None as parent if not defined
    start_node = Node(position=start)
    start_node.g = 0     # cost from start Node
    # heuristic estimated cost to end Node
    start_node.h = dist(start, end) if not use_manhattan else manhattan_dist(start, end)
    start_node.f = start_node.g + start_node.h

    end_node = Node(position=end)
    end_node.g = float('inf')      # set a large value if not defined
    end_node.h = 0       # heuristic estimated cost to end Node
    end_node.f = end_node.g + end_node.h

    # Initialize both yet_to_visit and visited dictionary
    # in this dict we will put all node that are yet_to_visit for exploration.
    # From here we will find the lowest cost node to expand next
    yet_to_visit_dict = {}  # key is the position (tuple), value is the node
    # in this list we will put all node those already explored so that we don't explore it again
    # key is the position (tuple), value is True (boolean)
    visited_dict = {}

    # Add the start node
    yet_to_visit_dict[start_node.position] = start_node

    # Adding a stop condition. This is to avoid any infinite loop and stop
    # execution after some reasonable number of steps
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

    # TODO PART 4 what squares do we search . serarch movement is left-right-top-bottom
    # (4 or 8 movements) from every positon
    # note that position is encoded as (x,y)
    move = [[0, 1],  # go up
            [-1, 0],  # go left
            [0, -1],  # go down
            [1, 0],  # go right
            [-1, 1],  # go up left
            [-1, -1],  # go down left
            [1, 1],  # go up right
            [1, -1]]  # go down right

    """
        1) We first get the current node by comparing all f cost and selecting the lowest cost node for further expansion
        2) Check max iteration reached or not . Set a message and stop execution
        3) Remove the selected node from yet_to_visit dict and add this node to visited dict
        4) Perofmr Goal test and return the path else perform below steps
        5) For selected node find out all children (use move to find children)
            a) get the current postion for the selected node (this becomes parent node for the children)
            b) check if a valid position exist (boundary will make few nodes invalid)
            c) if any node is a wall then ignore that
            d) add to valid children node list for the selected parent
            
            For all the children node
                a) if child in visited dict then ignore it and try next node
                b) calculate child node g, h and f values
                c) if child in yet_to_visit dict then ignore it
                d) else move the child to yet_to_visit dict
    """
    # TODO PART 4 find maze has got how many rows and columns
    no_rows, no_columns = np.shape(maze)

    # Loop until you find the end

    while len(yet_to_visit_dict) > 0:

        # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
        outer_iterations += 1

        # Get the current node with the lowest f value
        current_node = None
        current_fscore = None
        for position, node in yet_to_visit_dict.items():
            if current_fscore is None or node.f < current_fscore:
                current_fscore = node.f
                current_node = node

        # if we hit this point return the path such as it may be no solution or
        # computation cost is too high
        if outer_iterations > max_iterations:
            print("giving up on pathfinding too many iterations")
            return return_path(current_node, maze)

        # Pop current node out off yet_to_visit dict, add to visited list
        yet_to_visit_dict.pop(current_node.position)
        visited_dict[current_node.position] = True

        # test if goal is reached or not, if yes then return the path
        if current_node == end_node:

            return return_path(current_node, maze)

        # Generate children from all adjacent squares
        children = []

        for new_position in move:

            # TODO PART 4 Get node position
            node_position = (current_node.position[0] + new_position[0], 
                             current_node.position[1] + new_position[1])

            # TODO PART 4 Make sure within range (check if within maze boundary)
            if not (0 <= node_position[0] <= no_rows and 0 <= node_position[1] <= no_columns):
                continue

            # Make sure walkable terrain
            if maze[node_position[0], node_position[1]] > 0.8:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children

        for child in children:

            # TODO PART 4 Child is on the visited dict (use get method to check if child is in visited dict, if not found then default value is False)
            if child.position in visited_dict:
                continue

            # TODO PART 4 Create the f, g, and h values
            child.g = current_node.g + 1 # temp value modify later
            # Heuristic costs calculated here, this is using eucledian distance
            child.h = dist(child.position, end) if not use_manhattan else manhattan_dist(child.position, end)

            child.f = child.g + child.h

            # Child is already in the yet_to_visit list and g cost is already lower
            child_node_in_yet_to_visit = yet_to_visit_dict.get(
                child.position, False)
            if (child_node_in_yet_to_visit is not False) and (child.g >= child_node_in_yet_to_visit.g):
                continue

            # Add the child to the yet_to_visit list
            yet_to_visit_dict[child.position] = child
