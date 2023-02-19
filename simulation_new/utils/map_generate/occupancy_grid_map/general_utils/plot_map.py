#!~/anaconda3/envs/teb/bin/python
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from occupancy_grid_map.general_utils.gridmap import GridMapPath

def plot_path(gmap: GridMapPath, path, simplified_path=None, grid=False):
    """
    Plot the map and path (and the simplified path points if provided)
    :param gmap: grid map representing the environment
    :param path: continuous path points in meters
    :param simplified_path: down-sampled discrete path points in meters
    """
    fig_gridmap = plt.figure('grid_map_with_path')
    ax = fig_gridmap.add_axes([0.05,0.05,0.95,0.95], projection='3d')

    path_arr = np.array(path)
    ax.plot(path_arr[:,0], path_arr[:,1], path_arr[:,2], 'r--', label='path')
    _start = path[0]
    _goal = path[-1]
    ax.scatter([_start[0]], [_start[1]], [_start[2]], c='b', marker='o')
    ax.scatter([_goal[0]], [_goal[1]], [_goal[2]], c='k', marker='o')

    if simplified_path is not None:
        spath_arr = np.array(simplified_path)
        ax.plot(spath_arr[:,0], spath_arr[:,1], spath_arr[:,2], 'g:', label='simplified path')

    gmap.plot(axis=ax, grid=grid)

def plot_multi_surfaces(A, b, plot_range=[[0,5],[0,5],[0,5]], ax=None):
    """
    Plot multiple 3D surfaces defined by Ax=b
    :param A: (n*3 array) every row defines parameters of one surface
    :param b: (n*1 array) every element defines bias of one surface
    :param plot_range: (3*2 list) define plot range align x, y, z axes
    :param ax: matplotlib axis to draw surfaces
    """
    assert A.shape[0] == b.shape[0]
    if ax == None:
        figure = plt.figure('multiple surfaces')
        ax = figure.add_axes([0.05,0.05,0.95,0.95], projection='3d')
    
    x_crd = np.linspace(plot_range[0][0], plot_range[0][1], 10)
    y_crd = np.linspace(plot_range[1][0], plot_range[1][1], 10)
    z_crd = np.linspace(plot_range[2][0], plot_range[2][1], 10)
    for i in range(A.shape[0]):
        A_param = A[i]
        b_param = b[i]
        if A_param[2] != 0:
            x_grid, y_grid = np.meshgrid(x_crd, y_crd)
            z_grid = (b_param-A_param[0]*x_grid-A_param[1]*y_grid) / A_param[2]
        elif A_param[1] != 0:
            x_grid, z_grid = np.meshgrid(x_crd, z_crd)
            y_grid = (b_param-A_param[0]*x_grid-A_param[2]*z_grid) / A_param[1]
        elif A_param[0] != 0:
            y_grid, z_grid = np.meshgrid(y_crd, z_crd)
            x_grid = (b_param-A_param[1]*y_grid-A_param[2]*z_grid) / A_param[0]
        else:
            print("Invalid surface!")
            exit(-1)
        ax.plot_surface(x_grid, y_grid, z_grid, color=[0.4,0.4,0.4,0.3])

def plot_multi_ellipsoids(E, p, ax=None):
    """
    Plot multiple 3D surfaces defined by Ax=b
    :param E: (n*3*3 array) every 3*3 block defines parameters of one ellipsoid
    :param p: (n*3 array) every row defines bias of one ellipsoid
    :param ax: matplotlib axis to draw ellipsoids
    """
    assert len(E) == len(p)
    if ax == None:
        figure = plt.figure('multiple ellipsoids')
        ax = figure.add_axes([0.05,0.05,0.95,0.95], projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x_grid = np.outer(np.cos(u), np.sin(v))
    y_grid = np.outer(np.sin(u), np.sin(v))
    z_grid = np.outer(np.ones(np.size(u)), np.cos(v))
    for t in range(len(E)):
        E_param = E[t]
        p_param = p[t]
        new_xp = np.zeros([50,25])
        new_yp = np.zeros([50,25])
        new_zp = np.zeros([50,25])
        for i in range(50):
            for j in range(25):
                rot_vec = np.dot(E_param, np.array([x_grid[i][j],
                                                    y_grid[i][j],
                                                    z_grid[i][j]]).T) + p_param
                new_xp[i][j] = rot_vec[0]
                new_yp[i][j] = rot_vec[1]
                new_zp[i][j] = rot_vec[2]
        ax.plot_surface(new_xp, new_yp, new_zp, color=[0,1,0,0.3])