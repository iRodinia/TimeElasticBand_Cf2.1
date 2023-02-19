import numpy as np
import math
from .TrajGen.trajGen import trajGenerator
from simulation_new.utils.uav_trajectory.frenet import FrenetTrajectory
from simulation_new.utils.map_generate.occupancy_grid_map.general_utils.gridmap import GridMapPath
from .a_star import a_star
from .rrt_algorithm import rrt, rrt_connect, rrt_star, rrt_star_bid, rrt_star_bid_h
from simulation_new.utils.map_generate.occupancy_grid_map.general_utils.path_simplification import path_simplification

def point2line_dist(lp1, lp2, p):
    vec1 = lp1-p
    vec2 = lp2-p
    vec3 = lp2-lp1
    return np.linalg.norm(np.cross(vec1,vec2)) / np.linalg.norm(vec3)

def point2line_project(lp1, lp2, p):
    vec1 = p-lp1
    vec2 = lp2-lp1
    t = np.sum(vec1*vec2) / (np.linalg.norm(vec2)**2)
    return lp1 + t*vec2

def point2segment_dist(sp1, sp2, p):
    vec1 = p-sp1
    vec2 = sp2-sp1
    vec3 = p-sp2
    seg_length = np.linalg.norm(vec2)
    r = np.sum(vec1*vec2) / (seg_length*seg_length)
    if r <= 0:
        return np.linalg.norm(vec1)
    elif r >= 1:
        return np.linalg.norm(vec3)
    else:
        return np.linalg.norm(np.cross(vec1,vec3)) / seg_length

class TrajGenerator2:

    def __init__(self, map: GridMapPath, params):

        print("Initializing trajectory generator...")

        self.episode_len_sec = params["ctrl_time"]
        self.ctrl_freq = params["ctrl_freq"]
        self.ctrl_dt = 1.0 / self.ctrl_freq
        self.start_pos = params["start_pos"]
        self.stop_pos = params["stop_pos"][0:3]
        self.accuracy = params["accuracy"]
        self.uav_radius = params["uav_radius"]
        self.path_insert_point_dist_min = params["path_insert_point_dist_min"]
        self.traj_max_vel = params["traj_max_vel"]
        self.traj_gamma = params["traj_gamma"]

        self.map = map
        self.map.obstacle_dilation(dist=2*self.uav_radius)

        self._find_init_path()
        self._trajectory_plan()

    def _if_pos_collide_with_obstacles(self, pos):
        if pos[2] <= 0.03:
            return True
        xid, yid, zid = self.map.get_index_from_coordinates(pos[0], pos[1], pos[2])
        if self.map.is_inside_idx((xid, yid, zid)):
            if self.map.is_occupied_idx((xid, yid, zid)):
                return True
        return False

    def _find_init_path(self):
        start = (self.start_pos[0], self.start_pos[1], self.start_pos[2])
        stop = (self.stop_pos[0], self.stop_pos[1],self.stop_pos[2])
        # raw_path, _ = a_star(start, stop, self.map)
        raw_path = rrt_star(start, stop, self.map)
        s_path = path_simplification(self.map, raw_path)
        self.path = np.array(s_path)

        print("\033[0;32;40mInitial feasible path found!\033[0m")

    def _trajectory_plan(self):
        waypoints = self.path
        
        print("\033[7mConducting trajectory planning...\033[0m")

        self.minimum_snap = trajGenerator(np.array(waypoints), max_vel=self.traj_max_vel, gamma=self.traj_gamma)
        coef_mat = self.minimum_snap.get_coef_mat()
        self.trajectory = FrenetTrajectory(coef_mat=coef_mat, order=self.minimum_snap.order)

        timestamp = np.ones(int(self.ctrl_freq*self.episode_len_sec)) * self.ctrl_dt
        timestamp[0] = 0.0
        timestamp = np.cumsum(timestamp)
        self.timestamp = timestamp[timestamp<=self.minimum_snap.TS[-1]]
        self.pos_trajectory = np.array([self.minimum_snap.get_des_state(t).pos for t in timestamp])

        # temp = [self.trajectory.eval(t).pos for t in timestamp] # check if minimum_snap and trajectory get the same result

        print("\033[0;32;40mPlanning finished.\033[0m")
    
    def _intersect_point(self, p): # find a collision-free point from p that is close to self.path and insert that point to self.path
        dist = np.Inf
        index = 0
        mid_point = None
        safe_dist = 2*self.accuracy # delta = 0.1m for robustness
        for i in range(self.path.shape[0]-1):
            temp_dist = point2segment_dist(self.path[i], self.path[i+1], p)
            if temp_dist < dist:
                index = i
                dist = temp_dist
            if temp_dist <= 0.01:
                mid_point = p
                break
        if mid_point is None:
            project_point = point2line_project(self.path[index], self.path[index+1], p)
            direction = project_point - p
            direction_norm = np.linalg.norm(direction)
            if direction_norm >= safe_dist:
                mid_point = project_point
            else:
                mid_point = p + direction / direction_norm * safe_dist # TODO safety might not be guaranteed for this safe_dist
        
        for temp_p in self.path:
            if np.linalg.norm(mid_point - temp_p) <= self.path_insert_point_dist_min:
                return 0
        self.path = np.insert(self.path, index+1, mid_point, axis=0)
        return 1
    
    def trajectory_replan(self):
        collide_flag = False
        in_pos = []
        out_pos = []

        for p in self.pos_trajectory:
            if collide_flag is False:
                if self._if_pos_collide_with_obstacles(p):
                    in_pos.append(p)
                    collide_flag = True
            else:
                if not self._if_pos_collide_with_obstacles(p):
                    out_pos.append(p)
                    collide_flag = False
        
        added_point = 0
        for i in range(len(out_pos)):
            added_point += self._intersect_point((in_pos[i]+out_pos[i])/2)
        if added_point != 0:
            self._trajectory_plan()
            return True
        else:
            return False
