#!~/anaconda3/envs/teb/bin/python
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
from occupancy_grid_map.general_utils.gridmap import GridMapPath

def path_simplification(gmap: GridMapPath, path):
    """
    Path simplification for a 3D grid map:
    Find the discrete path points that in effect.

    :param gmap: GridMapPath that store the map parameters
    :param path: path points in meters

    :return: modified_path, discrete point list in meters
    """
    visited = np.zeros(gmap.dim_cells)

    def _is_intersect(p1, p2):
        point1 = np.array(p1)
        point2 = np.array(p2)
        direct = (point2 - point1) / np.linalg.norm(point1-point2)
        _d = gmap.cell_size
        num = int(round(np.linalg.norm(point2 - point1) / _d))
        for k in range(num):
            temp_p = point1 + k*_d*direct
            x, y, z = gmap.get_index_from_coordinates(temp_p[0], temp_p[1], temp_p[2])
            if visited[x][y][z] == 1.:
                continue
            if gmap.data[x][y][z] == 1:
                return True
            visited[x][y][z] = 1.
        return False

    sp_path = [path[0]]
    for i in range(len(path)-1):
        begin = sp_path[-1]
        end = path[i+1]
        if _is_intersect(begin, end):
            sp_path.append(path[i])
    sp_path.append(path[-1])
    return sp_path