#!~/anaconda3/envs/teb/bin/python
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
import pybullet as p
from .gridmap import OccupancyGridMap3D

def _add_box_to_map(map: OccupancyGridMap3D, box_center, half_extend, rotation, d=0.02):
    rotmat = np.array(p.getMatrixFromQuaternion(rotation)).reshape((3,3))
    xmin = - half_extend[0]
    xmax = half_extend[0]
    ymin = - half_extend[1]
    ymax = half_extend[1]
    zmin = - half_extend[2]
    zmax = half_extend[2]

    for i in np.linspace(xmin, xmax, int(round((xmax-xmin)/d))+1):
        for j in np.linspace(ymin, ymax, int(round((ymax-ymin)/d))+1):
            for k in np.linspace(zmin, zmax, int(round((zmax-zmin)/d))+1):
                rot_vec = rotmat.dot(np.array([i,j,k]).T)
                vec = rot_vec + np.array(box_center)
                indx, indy, indz = map.get_index_from_coordinates(vec[0], vec[1], vec[2])
                if map.is_inside_idx((indx, indy, indz)):
                    map.data[indx][indy][indz] = 1

def _add_cylinder_to_map(map: OccupancyGridMap3D, cld_center, radius, height, rotation, d=0.05):
    rotmat = np.array(p.getMatrixFromQuaternion(rotation)).reshape((3,3))
    hmin = - height / 2
    hmax = height / 2

    for i in np.linspace(hmin, hmax, int(round((hmax-hmin)/d))+1):
        for j in np.linspace(0, radius, int(round(radius/d))+1):
            for k in np.linspace(0, 2*np.pi, 180, endpoint=False):
                x = j * np.cos(k)
                y = j * np.sin(k)
                z = i
                rot_vec = rotmat.dot(np.array([x,y,z]).T)
                vec = rot_vec + np.array(cld_center)
                indx, indy, indz = map.get_index_from_coordinates(vec[0],vec[1],vec[2])
                if map.is_inside_idx((indx, indy, indz)):
                    map.data[indx][indy][indz] = 1

def _add_ball_to_map(map: OccupancyGridMap3D, ball_center, radius, d=0.05):
    for i in np.linspace(0, radius, int(round(radius/d))+1):
        for j in np.linspace(0, 2*np.pi, 180, endpoint=False):
            for k in np.linspace(0, np.pi, 91):
                x = i * np.sin(k) * np.cos(j) + ball_center[0]
                y = i * np.sin(k) * np.sin(j) + ball_center[1]
                z = i * np.cos(k) + ball_center[2]
                indx, indy, indz = map.get_index_from_coordinates(x,y,z)
                if map.is_inside_idx((indx, indy, indz)):
                    map.data[indx][indy][indz] = 1

def add_obstacles_to_gridmap(map: OccupancyGridMap3D, obstacles: dict={}):
    """
    Load box and cylinder obstacles into the grid map
    :param pclient: pybullet client
    :param obstacles: dict of obstacles, temporarily support:
        {type: "box", center(1x3 array), half_extend(1x3 array), yaw_rad, fixed(bool)}
        {type: "cylinder", center(1x3 array), radius, height, fixed(bool)}
        {type: "ball", center(1x3 array), radius, fixed(bool)}
    """
    for key in obstacles:
        obs = obstacles[key]
        if obs['type'] == "box":
            rot = p.getQuaternionFromEuler([0, 0, obs['yaw_rad']])
            _add_box_to_map(map, obs['center'], obs['half_extend'], rot)
        elif obs['type'] == "cylinder":
            _add_cylinder_to_map(map, obs['center'], obs['radius'], obs['height'], [0,0,0,1])
        elif obs['type'] == "ball":
            _add_ball_to_map(map, obs['center'], obs['radius'])
        else:
            pass

if __name__ == '__main__':
    from occupancy_grid_map.general_utils.gridmap import GridMapFromImage

    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/maps/simple.png'
    grid = GridMapFromImage(path, 2., cell_size=0.1, with_path=False)
    _add_box_to_map(grid, [1.1,0.5,0.6], [0.2,0.2,0.6],
                    p.getQuaternionFromEuler([0, 0, 1.58]), 0.05)
    _add_cylinder_to_map(grid, [0.25,2.3,0.7], 0.25, 0.7,
                    p.getQuaternionFromEuler([0, 0, 0]), 0.05)
    _add_ball_to_map(grid, [1.75, 1.6, 0.8], 0.3, 0.05)
    
    grid.plot(grid=True)