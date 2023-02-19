#!~/anaconda3/envs/teb/bin/python
import os

import numpy as np
from scipy import sparse
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib.image as mpimg

class OccupancyGridMap3D:

    def __init__(self, array3D: np.ndarray, crd_bias=[0,0,0], cell_size=.05, occupancy_threshold=.5):
        """
        Creates a grid map from an image
        :param array3D: the 3D occupancy array (X, Y, Z)
        :param cell_size: cell size in meters
        :param occupancy_threshold: A threshold to determine whether a cell is occupied or free.
        A cell is considered occupied if its value >= occupancy_threshold, free otherwise.
        """
        self.crd_bias = np.array(crd_bias)
        self.data = array3D
        self.dim_cells = self.data.shape
        self.dim_meters = cell_size * np.array(self.dim_cells)
        self.cell_size = cell_size
        self.occupancy_threshold = occupancy_threshold

    def is_occupied_idx(self, point_idx):
        x_index, y_index, z_index = point_idx
        if self.data[x_index][y_index][z_index] >= self.occupancy_threshold:
            return True
        else:
            return False
    
    def is_inside_idx(self, point_idx):
        x_index, y_index, z_index = point_idx
        if x_index < 0 or y_index < 0 or z_index < 0 \
            or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1] or z_index >= self.dim_cells[2]:
            return False
        return True
    
    def set_threshold(self, threshold: float):
        if threshold >= 0 and threshold <= 1:
            self.occupancy_threshold = threshold
            return True
        return False

    def get_index_from_coordinates(self, x, y, z):
        x = x - self.crd_bias[0]
        y = y - self.crd_bias[1]
        z = z - self.crd_bias[2]
        if x == self.dim_meters[0]:
            x_index = int(x/self.cell_size) - 1
        else:
            x_index = int(x/self.cell_size)
        if y == self.dim_meters[1]:
            y_index = int(y/self.cell_size) - 1
        else:
            y_index = int(y/self.cell_size)
        if z == self.dim_meters[2]:
            z_index = int(z/self.cell_size) - 1
        else:
            z_index = int(z/self.cell_size)

        return x_index, y_index, z_index

    def get_coordinates_from_index(self, x_index, y_index, z_index):
        x = x_index*self.cell_size
        y = y_index*self.cell_size
        z = z_index*self.cell_size
        x = x + self.crd_bias[0]
        y = y + self.crd_bias[1]
        z = z + self.crd_bias[2]

        return x, y, z
    
    def map_normalize(self):
        self.data = np.clip(np.around(self.data), 0., 1.)
    
    def obstacle_dilation(self, dist=0.):
        half_len = max(1, int(round(dist/self.cell_size)))

        conv_core = np.ones((2*half_len+1, 2*half_len+1, 2*half_len+1))
        new_data = signal.convolve(self.data, conv_core, mode='same')
        self.data = new_data
        self.map_normalize()
        
    def plot(self, axis=None, grid=False):
        """
        plot the grid map
        """
        if axis == None:
            figure = plt.figure('grid_map')
            axis = figure.add_axes([0.05,0.05,0.95,0.95], projection='3d')
        ax = axis
        if grid:
            color = mpc.to_rgba('lightseagreen', alpha=0.4)
            vs = self.data
            x,y,z = np.indices(np.array(vs.shape)+1)
            ax.voxels(x*self.cell_size, y*self.cell_size, z*self.cell_size, vs, facecolors=color)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('3D Voxel Map')
        else:
            print("Warning: This function is only suitable for drawing the obstacles that are not hollowed out!")
            # true_data = np.clip(np.around(self.data), 0., 1.)
            Z = self.data.sum(axis=2) * self.cell_size
            X, Y = np.meshgrid(np.linspace(0,self.dim_meters[0], num=self.dim_cells[0]),
                                np.linspace(0,self.dim_meters[1], num=self.dim_cells[1]))
            ax.contour(X, Y, Z.T, 3, extend3d=True, colors='darkgrey', alpha=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('3D Contour')

        plt.show(block=False)
        plt.pause(5)
        plt.close()

class GridMapPath(OccupancyGridMap3D):

    def __init__(self, array3D, crd_bias=[0,0,0], cell_size=.1, occupancy_threshold=.5):
        super().__init__(array3D, crd_bias, cell_size, occupancy_threshold)
        self.visited = np.zeros(self.dim_cells, dtype=np.float32)
    
    def get_data_idx(self, point_idx):
        x_index, y_index, z_index = point_idx
        if x_index < 0 or y_index < 0 or z_index < 0 \
            or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1] or z_index >= self.dim_cells[2]:
            raise Exception('Point is outside map boundary')
        return self.data[x_index][y_index][z_index]

    def mark_visited_idx(self, point_idx):
        x_index, y_index, z_index = point_idx
        if x_index < 0 or y_index < 0 or z_index < 0 \
            or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1] or z_index >= self.dim_cells[2]:
            raise Exception('Point is outside map boundary')
        self.visited[x_index][y_index][z_index] = 1.0

    def is_visited_idx(self, point_idx):
        x_index, y_index, z_index = point_idx
        if x_index < 0 or y_index < 0 or z_index < 0 \
            or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1] or z_index >= self.dim_cells[2]:
            raise Exception('Point is outside map boundary')
        if self.visited[x_index][y_index][z_index] == 1.0:
            return True
        else:
            return False

def GridMapFromImage(imgpath: str, height: float, cell_size: float=0.1, with_path: bool=True):
    img_array = mpimg.imread(imgpath)
    gray_img = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            if gray_img[i][j] >= 0.5:
                gray_img[i][j] = 1
            else:
                gray_img[i][j] = 0
    height_num = int(round(height / cell_size))
    _data = np.broadcast_to(gray_img.reshape((gray_img.shape[0], gray_img.shape[1], 1)),
                                (gray_img.shape[0], gray_img.shape[1], height_num))
    if not with_path:
        return OccupancyGridMap3D(_data.copy(), cell_size=cell_size)
    else:
        return GridMapPath(_data.copy(), cell_size=cell_size)

def BlankGridMap(length: float, width: float, height: float, cell_size: float=0.05, with_path: bool=True):
    len_dim = int(round(length / cell_size))
    wid_dim = int(round(width / cell_size))
    hgt_dim = int(round(height / cell_size))
    _data = np.zeros((len_dim,wid_dim,hgt_dim), dtype=int)
    if not with_path:
        return OccupancyGridMap3D(_data.copy(), cell_size=cell_size)
    else:
        return GridMapPath(_data.copy(), cell_size=cell_size)

if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + '/pictures/test.png'
    grid = GridMapFromImage(path, 3.)
    grid.obstacle_dilation(0.2)
    grid.plot(grid=False)