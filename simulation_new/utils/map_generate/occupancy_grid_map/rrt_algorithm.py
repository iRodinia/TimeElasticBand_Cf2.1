#!~/anaconda3/envs/teb/bin/python
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
from occupancy_grid_map.general_utils.gridmap import GridMapPath, GridMapFromImage
from occupancy_grid_map.rrt.rrt_utils.search_space import SearchSpace

from occupancy_grid_map.rrt.rrt import RRT
from occupancy_grid_map.rrt.rrt_connect import RRTConnect
from occupancy_grid_map.rrt.rrt_star import RRTStar
from occupancy_grid_map.rrt.rrt_star_bid import RRTStarBidirectional
from occupancy_grid_map.rrt.rrt_star_bid_h import RRTStarBidirectionalHeuristic

def rrt(start_m, goal_m, gmap: GridMapPath):
    space = SearchSpace(gmap)
    search_dist = gmap.cell_size
    Q = np.array([(search_dist, 5)])
    r = gmap.cell_size / 2
    max_samples = 10000
    prc = 0.01

    _rrt = RRT(space, Q, start_m, goal_m, max_samples, r, prc)
    path_m = _rrt.rrt_search()
    return path_m

def rrt_star(start_m, goal_m, gmap: GridMapPath):
    space = SearchSpace(gmap)
    search_dist = gmap.cell_size
    Q = np.array([(search_dist, 5)])
    r = gmap.cell_size / 2
    max_samples = 10000
    prc = 0.01
    rewire_num = 32

    _rrt_star = RRTStar(space, Q, start_m, goal_m, max_samples, r, prc, rewire_num)
    path_m = _rrt_star.rrt_star()
    return path_m

def rrt_connect(start_m, goal_m, gmap: GridMapPath):
    space = SearchSpace(gmap)
    search_dist = gmap.cell_size * 2
    Q = np.array([search_dist])
    r = gmap.cell_size
    max_samples = 10000
    prc = 0.01

    _rrt_connect = RRTConnect(space, Q, start_m, goal_m, max_samples, r, prc)
    path_m = _rrt_connect.rrt_connect()
    return path_m

def rrt_star_bid(start_m, goal_m, gmap: GridMapPath):
    space = SearchSpace(gmap)
    search_dist = gmap.cell_size
    Q = np.array([(search_dist, 5)])
    r = gmap.cell_size
    max_samples = 10000
    prc = 0.01
    rewire_num = 32

    _rrt_star_bid = RRTStarBidirectional(space, Q, start_m, goal_m, max_samples, r, prc, rewire_num)
    path_m = _rrt_star_bid.rrt_star_bidirectional()
    return path_m

def rrt_star_bid_h(start_m, goal_m, gmap: GridMapPath):
    space = SearchSpace(gmap)
    search_dist = gmap.cell_size
    Q = np.array([(search_dist, 5)])
    r = gmap.cell_size
    max_samples = 10000
    prc = 0.01
    rewire_num = 32

    _rrt_star_bid_h = RRTStarBidirectionalHeuristic(space, Q, start_m, goal_m, max_samples, r, prc, rewire_num)
    path_m = _rrt_star_bid_h.rrt_star_bidirectional()
    return path_m

if __name__ == '__main__':
    from occupancy_grid_map.general_utils.plot_map import plot_path
    from occupancy_grid_map.general_utils.path_simplification import path_simplification

    path = os.path.abspath(os.path.dirname(__file__)) + '/maps/simple.png'
    grid = GridMapFromImage(path, 1.5, cell_size=0.1, with_path=False)
    start = (0.2, 0.2, 0.2)
    goal = (0.8, 0.2, 1.0)
    path = rrt_star(start, goal, grid)
    discrete_path = path_simplification(grid, path)

    plot_path(grid, path, discrete_path, grid=True)