#!~/anaconda3/envs/teb/bin/python
import os
import sys
sys.path.append(os.path.abspath('.'))

import math
import numpy as np
from heapq import heappush, heappop
from simulation.utils.map_generate.occupancy_grid_map.general_utils.gridmap import GridMapPath, GridMapFromImage

def dist3d(point1, point2):
    x1, y1, z1 = point1[0:3]
    x2, y2, z2 = point2[0:3]
    dist = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
    return math.sqrt(dist)

def _get_movements_6n():
    """
    Get all possible 6-connectivity movements.
    Movements along z axis are a little bit harder
    :return: list of movements with cost [(dx, dy, dz, movement_cost)]
    """
    return [(1, 0, 0, 1.0),
            (0, 1, 0, 1.0),
            (0, 0, 1, 1.1),
            (-1, 0, 0, 1.0),
            (0, -1, 0, 1.0),
            (0, 0, -1, 1.1)]

def a_star(start_m, goal_m, gmap: GridMapPath):
    """
    A* for 3D occupancy grid.
    Movement allowed '6N': front, back, left, right, up, down

    :param start_m: start node (x, y, z) in meters
    :param goal_m: goal node (x, y, z) in meters
    :param gmap: the grid map

    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """
    # get array indices of start and goal
    start = gmap.get_index_from_coordinates(start_m[0], start_m[1], start_m[2])
    goal = gmap.get_index_from_coordinates(goal_m[0], goal_m[1], goal_m[2])

    # check if start and goal nodes correspond to free spaces
    if gmap.is_occupied_idx(start):
        raise Exception('Start node is not traversable')

    if gmap.is_occupied_idx(goal):
        raise Exception('Goal node is not traversable')

    # add start node to front
    # front is a list of (total estimated cost to goal, total cost from start to node, node, previous node)
    start_node_cost = 0
    start_node_estimated_cost_to_goal = dist3d(start, goal) + start_node_cost
    front = [(start_node_estimated_cost_to_goal, start_node_cost, start, None)]

    # use a dictionary to remember where we came from in order to reconstruct the path later on
    came_from = {}

    # get possible movements
    movements = _get_movements_6n()

    # while there are elements to investigate in our front.
    while front:
        # get smallest item and remove from front.
        element = heappop(front)

        # if this has been visited already, skip it
        _, from_cost, pos, previous = element
        if gmap.is_visited_idx(pos):
            continue

        # now it has been visited, mark with cost
        gmap.mark_visited_idx(pos)

        # set its previous node
        came_from[pos] = previous

        # if the goal has been reached, we are done!
        if pos == goal:
            break

        # check all neighbors
        for dx, dy, dz, deltacost in movements:
            # determine new position
            new_x = pos[0] + dx
            new_y = pos[1] + dy
            new_z = pos[2] + dz
            new_pos = (new_x, new_y, new_z)

            # check whether new position is inside the map
            # if not, skip node
            if not gmap.is_inside_idx(new_pos):
                continue

            # add node to front if it was not visited before and is not an obstacle
            if (not gmap.is_visited_idx(new_pos)) and (not gmap.is_occupied_idx(new_pos)):
                new_cost = from_cost + deltacost
                new_total_cost_to_goal = new_cost + dist3d(new_pos, goal)

                heappush(front, (new_total_cost_to_goal, new_cost, new_pos, pos))

    # reconstruct path backwards (only if we reached the goal)
    path = []
    path_idx = []
    if pos == goal:
        while pos:
            path_idx.append(pos)
            # transform array indices to meters
            pos_m_x, pos_m_y, pos_m_z = gmap.get_coordinates_from_index(pos[0], pos[1], pos[2])
            path.append((pos_m_x, pos_m_y, pos_m_z))
            pos = came_from[pos]

        # reverse so that path is from start to goal.
        path.reverse()
        path_idx.reverse()

    return path, path_idx

if __name__ == '__main__':
    from occupancy_grid_map.general_utils.plot_map import plot_path
    from occupancy_grid_map.general_utils.path_simplification import path_simplification

    path = os.path.abspath(os.path.dirname(__file__)) + '/maps/test.png'
    grid = GridMapFromImage(path, 3., cell_size=0.1, with_path=True)
    start = [0.3, 4.5, 0.2]
    goal = [2.5, 4.5, 2.4]
    path, path_idx = a_star(start, goal, grid)
    
    s_path = path_simplification(grid, path)
    plot_path(grid, path, s_path)