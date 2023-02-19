#!~/anaconda3/envs/teb/bin/python
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
from occupancy_grid_map.general_utils.gridmap import OccupancyGridMap3D

def rodrigues(vec1, vec2):
    """
    Using Rodrigues formula to calculate the rotation matrix M
    vec2 = M * vec1
    :param vec1: (1*3 array) the vector to rotate from
    :param vec2: (1*3 array) the vector to rotate to
    """
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return np.eye(3)
    v1 = np.array(vec1)/n1
    v2 = np.array(vec2)/n2
    c = np.sum(v1*v2)
    v = np.cross(v1, v2)
    if np.linalg.norm(v) == 0 and c == -1:
        return -np.eye(3)
    v_x = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    return np.eye(3) + v_x + (1/(1+c))*v_x.dot(v_x)

def tangent_plane_of_ellipsoid(E, d, point):
    """
    Calculate the tangent plane of ellipsoid at the point.
    Ellipsoid: p = Ex + d, ||x||=1, E is positive-define symmetric matrix
    :param E: (3*3 array) the ellipsoid matrix, E=R*S*R.T, S=diag([a,b,c]), R is the rotation matrix
    :param d: (1*3 array) the center of the ellipsoid
    :param point: (1*3 array) the tangent point on the ellipsoid
    Return the tangent plane formulation A, b satisfy Ax=b
    The ellipsoid satisfy Ax<=b
    """
    E_inv = np.linalg.inv(np.array(E))
    A = 2 * np.dot(np.dot(E_inv, E_inv), (point-d).T)
    b = np.sum(A * point)
    return A, b

# test only
# points = []
# E_s = []
# p_s = []

class ObstacleMonitor:
    def __init__(self, box_center, box_rot_mat, collision_fcn, corridor_halflen, ds=0.01):
        """
        A grid array to monitor and modify obstacles, provide utility interface to ConvexHullGenerator
        :param box_center: (1*3 array) the center coordinate of the box corridor in the world frame
        :param box_rot_mat: (3*3 array) the rotation matrix of the box corridor to the world frame
        :param collision_fcn: (function handle) collision_fcn(point) == True if point encounters 
            with the obstacles otherwise False, point is of the world coordinate
        :param corridor_halflen: (1*3 array) the half extend length of the corridor box
        :param ds: (float) accuracy of the computation process in meters
        """
        self.corridor_halflen = corridor_halflen
        self.array_dim = [2*int(round(corridor_halflen[0]/ds))+1,
                            2*int(round(corridor_halflen[1]/ds))+1,
                            2*int(round(corridor_halflen[2]/ds))+1]
        self.collision_fcn = collision_fcn
        self.ds = ds
        self.center_crd = np.array(box_center)
        self.rot_mat = box_rot_mat
        self.array = self._get_array_from_fcn(collision_fcn)
    
    def _get_array_from_fcn(self, fcn):
        data = np.zeros(self.array_dim)
        for i in range(self.array_dim[0]):
            for j in range(self.array_dim[1]):
                for k in range(self.array_dim[2]):
                    crd = self.from_idx_to_crd([i, j, k])
                    if fcn(crd):
                        data[i][j][k] = 1
        return data

    def _get_center_idx(self):
        return [int((self.array_dim[0]-1)/2),
                int((self.array_dim[1]-1)/2),
                int((self.array_dim[2]-1)/2)]
    
    def is_inside_idx(self, idx):
        assert len(idx) == 3
        if idx[0] < 0 or idx[0] >= self.array_dim[0] or\
            idx[1] < 0 or idx[1] >= self.array_dim[1] or\
            idx[2] < 0 or idx[2] >= self.array_dim[2]:
            return False
        return True
    
    def is_inside_crd(self, crd):
        assert len(crd) == 3
        bias = self.rot_mat.T.dot((np.array(crd) - self.center_crd).T)
        if abs(bias[0]) > self.corridor_halflen[0] or\
            abs(bias[1]) > self.corridor_halflen[1] or\
            abs(bias[2]) > self.corridor_halflen[2]:
            return False
        return True
    
    def from_idx_to_crd(self, idx):
        assert len(idx) == 3
        assert self.is_inside_idx(idx)
        dist_idx = np.array(idx) - np.array(self._get_center_idx())
        dist_crd = dist_idx*self.ds
        rot_dist = self.rot_mat.dot(dist_crd.T)
        crd = self.center_crd + rot_dist
        return crd
    
    def from_crd_to_idx(self, crd):
        assert len(crd) == 3
        # assert self.is_inside_crd(crd)
        bias = self.rot_mat.T.dot((np.array(crd) - self.center_crd).T)
        dist_idx = self._get_center_idx() + np.array([int(bias[0]/self.ds),
                                                        int(bias[1]/self.ds),
                                                        int(bias[2]/self.ds)])
        dist_idx[0] = max(min(dist_idx[0],self.array_dim[0]-1),0)
        dist_idx[1] = max(min(dist_idx[1],self.array_dim[1]-1),0)
        dist_idx[2] = max(min(dist_idx[2],self.array_dim[2]-1),0)
        return dist_idx
    
    def is_collide_crd(self, crd):
        if self.is_inside_crd(crd):
            idx = self.from_crd_to_idx(crd)
            if self.array[idx[0]][idx[1]][idx[2]] >= 1:
                return True
        return False
    
    def is_free_space(self):
        if self.array.max() < 1.:
            return True
        return False
    
    def exclude_obstacles(self, A, b, center_point):
        """
        Exclude obstacles that satisfy Ax>=b in the world frame
        :param A: (3*3 ndarray) the map matrix
        :param b: (3*1 ndarray) the bounding condition
        :param center_point: (1*3 array) the collide point coordinate
        TODO: simplify calculation
        """
        # idx = self.from_crd_to_idx(center_point)
        for i in range(self.array_dim[0]):
            for j in range(self.array_dim[1]):
                for k in range(self.array_dim[2]):
                    if self.array[i][j][k] < 1.:
                        continue
                    crd = self.from_idx_to_crd([i, j, k])
                    if np.linalg.norm(crd - center_point) <= 8*self.ds:
                    # TODO: consider use clustering method to re-establish this method
                        self.array[i][j][k] = 0
                    t = A.dot(crd.T)
                    if t >= b:
                        self.array[i][j][k] = 0


class ConvexHullGenerator:
    def __init__(self, start, stop, collision_fcn, corridor_halflen, ds=0.01):
        """
        Generate convex hull constraints for one line segment from environment
        :param start: (1*3 array) the start position
        :param stop: (1*3 array) the stop position
        :param collision_fcn: (function handle) collision_fcn(point) == True if point encounters 
            with the obstacles otherwise False
        :param corridor_halflen: (1*3 array) the half extend length of the corridor box
        """
        self.start = np.array(start)
        self.stop = np.array(stop)
        self.mid_point = (self.start + self.stop) / 2
        self.corridor_halflen = corridor_halflen
        self.rot_mat = rodrigues([1,0,0], self.stop-self.start)
        self.monitor = ObstacleMonitor(self.mid_point, self.rot_mat,
                                        collision_fcn, corridor_halflen, ds)
        self.ds = ds
    
    def get_corridor_constraints(self, add_front_and_tail=True):
        A = []
        b = []
        rot_bias = np.dot(self.rot_mat.T, self.mid_point.T)
        A.append(np.dot(np.array([0,1,0]), self.rot_mat.T))
        b.append(self.corridor_halflen[1] + np.dot(np.array([0,1,0]),rot_bias))
        A.append(np.dot(np.array([0,-1,0]), self.rot_mat.T))
        b.append(self.corridor_halflen[1] + np.dot(np.array([0,-1,0]),rot_bias))
        A.append(np.dot(np.array([0,0,1]), self.rot_mat.T))
        b.append(self.corridor_halflen[2] + np.dot(np.array([0,0,1]),rot_bias))
        A.append(np.dot(np.array([0,0,-1]), self.rot_mat.T))
        b.append(self.corridor_halflen[2] + np.dot(np.array([0,0,-1]),rot_bias))
        if add_front_and_tail:
            A.append(np.dot(np.array([1,0,0]), self.rot_mat.T))
            b.append(self.corridor_halflen[0] + np.dot(np.array([1,0,0]),rot_bias))
            A.append(np.dot(np.array([-1,0,0]), self.rot_mat.T))
            b.append(self.corridor_halflen[0] + np.dot(np.array([-1,0,0]),rot_bias))
        return A, b
    
    def get_convex_constraints(self, max_iter=500):
        xlen = np.linalg.norm(self.stop-self.start) / 2
        u_num = 50
        v_num = 25
        u = np.linspace(0, 2 * np.pi, u_num)
        v = np.linspace(0, np.pi, v_num)
        x_grid = np.outer(np.cos(u), np.sin(v))
        y_grid = np.outer(np.sin(u), np.sin(v))
        z_grid = np.outer(np.ones(np.size(u)), np.cos(v))
        A = []
        b = []
        for iter_num in range(max_iter):
            if self.monitor.is_free_space():
                return A, b
            ellipsoid_radius = (iter_num+1) * self.ds
            S = np.array([[xlen,0,0],
                        [0,ellipsoid_radius,0],
                        [0,0,ellipsoid_radius]])
            E = np.dot(np.dot(self.rot_mat, S), self.rot_mat.T)
            for i in range(u_num):
                for j in range(v_num):
                    rot_vec = np.dot(E, np.array([x_grid[i][j], y_grid[i][j], z_grid[i][j]]).T)
                    exam_point = rot_vec + self.mid_point
                    if self.monitor.is_collide_crd(exam_point):

                        # test only
                        # points.append(exam_point)
                        # E_s.append(E)
                        # p_s.append(self.mid_point)

                        A_temp, b_temp = tangent_plane_of_ellipsoid(E, self.mid_point, exam_point)
                        A.append(A_temp)
                        b.append(b_temp)
                        self.monitor.exclude_obstacles(A_temp, b_temp, exam_point)
        return A, b

    def get_constraints(self):
        """
        Generate convex hull constraints as Ax<=b
        """
        A_cor, b_cor = self.get_corridor_constraints()
        A_con, b_con = self.get_convex_constraints()
        if len(A_con) == 0:
            return np.array(A_cor), np.array(b_cor)
        return np.concatenate((A_cor, A_con)), np.concatenate((b_cor, b_con))

def collision_fcn_from_gridmap(map: OccupancyGridMap3D):
    def fcn_bar(crd):
        xid, yid, zid = map.get_index_from_coordinates(crd[0], crd[1], crd[2])
        if map.is_inside_idx((xid, yid, zid)):
            return map.is_occupied_idx((xid, yid, zid))
        return False
    return fcn_bar

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from occupancy_grid_map.general_utils.gridmap import GridMapFromImage
    from occupancy_grid_map.general_utils.plot_map import plot_multi_surfaces, plot_multi_ellipsoids

    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/occupancy_grid_map/maps/test.png'
    grid = GridMapFromImage(path, 2., cell_size=0.1, with_path=False)
    collision_fcn = collision_fcn_from_gridmap(grid)

    fig = plt.figure('surfaces')
    ax = fig.add_axes([0.05,0.05,0.95,0.95], projection='3d')

    start = [1., 1., 0.6]
    stop = [2.4, 2., 1.3]
    ax.scatter(start[0], start[1], start[2], marker='v')
    ax.scatter(stop[0], stop[1], stop[2], marker='s')

    alen = np.linalg.norm(np.array(stop) - np.array(start)) / 2 + 0.2
    half_extend = [alen, 0.5, 0.5]
    hull_generator = ConvexHullGenerator(start, stop, collision_fcn, half_extend, ds=0.02)
    
    A, b = hull_generator.get_constraints()
    num_to_plot = 3
    # points = np.array(points)
    # E_s = np.array(E_s)
    # p_s = np.array(p_s)
    # # ax.scatter(points[num_to_plot-1,0], points[num_to_plot-1,1], points[num_to_plot-1,2], marker='*')
    # ax.scatter(points[:,0], points[:,1], points[:,2], marker='*', c='k')
    # plot_multi_ellipsoids([E_s[num_to_plot-1]], [p_s[num_to_plot-1]], ax)

    A_sub = A[num_to_plot-1+6,:]
    b_sub = b[num_to_plot-1+6]

    plot_multi_surfaces(np.array([A_sub]), np.array([b_sub]), [[1,3],[2,4],[0,2]], ax)

    ax.set_zlim(0, 4.)
    grid.plot(axis=ax, grid=False)