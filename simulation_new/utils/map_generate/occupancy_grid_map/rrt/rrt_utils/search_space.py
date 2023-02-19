#!~/anaconda3/envs/teb/bin/python
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
from occupancy_grid_map.general_utils.gridmap import GridMapPath

class SearchSpace(object):
    def __init__(self, gmap: GridMapPath):
        """
        Initialize Search Space
        :param gmap: grid map representing the environment
        """
        self.map = gmap

    def obstacle_free(self, x):
        """
        :param x: point in meters
        """
        assert len(x) == 3
        x_d, y_d, z_d = self.map.get_index_from_coordinates(x[0], x[1], x[2])
        if self.map.is_inside_idx((x_d, y_d, z_d)):
            if not self.map.is_occupied_idx((x_d, y_d, z_d)):
                return True
        return False

    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free in meters
        """
        while True:  # sample until not inside of an obstacle
            x = self.sample()
            if self.obstacle_free(x):
                return x

    def collision_free(self, start, end):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        visited = np.zeros(self.map.dim_cells)
        point1 = np.array(start)
        point2 = np.array(end)
        direct = (point2 - point1) / np.linalg.norm(point1 - point2)
        _d = self.map.cell_size
        num = int(round(np.linalg.norm(point2 - point1) / _d))
        for k in range(num):
            temp_p = point1 + k*_d*direct
            x, y, z = self.map.get_index_from_coordinates(temp_p[0], temp_p[1], temp_p[2])
            if visited[x][y][z] == 1.:
                continue
            if self.map.data[x][y][z] == 1:
                return False
            visited[x][y][z] = 1.
        return True

    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        dim = self.map.dim_cells
        _d = self.map.cell_size
        x = np.random.randint(0, dim[0]+1) * _d
        y = np.random.randint(0, dim[1]+1) * _d
        z = np.random.randint(0, dim[2]+1) * _d
        return (x, y, z)
    
    def project_to_bound(self, point):
        """
        Project point to the bound if it exceeds the space's boundary
        :param point: tuple, point coordinates in meters
        :return: boundary point if projected otherwise the original point
        """
        bound = self.map.dim_meters
        point = np.maximum(point, (0,0,0))
        point = np.minimum(point, bound)
        return tuple(point)