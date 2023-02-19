#!~/anaconda3/envs/teb/bin/python
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
from occupancy_grid_map.general_utils.gridmap import GridMapPath

def path_to_path_idx(gmap: GridMapPath, path):
    visited = np.zeros(gmap.dim_cells)
    path_idx = []
    for p in path:
        xi, yi, zi = gmap.get_index_from_coordinates(p[0], p[1], p[2])
        if visited[xi][yi][zi] == 0:
            path_idx.append((xi, yi, zi))
            visited[xi][yi][zi] = 1
    return path_idx