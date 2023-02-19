import numpy as np
import math
from .TrajGen.trajGen import trajGenerator
from simulation_new.utils.map_generate.occupancy_grid_map.general_utils.gridmap import OccupancyGridMap3D

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


class TrajGenerator1:

    def __init__(self, map: OccupancyGridMap3D, params):

        print("Initializing trajectory generator...")

        self.episode_len_sec = params["ctrl_time"]
        self.ctrl_freq = params["ctrl_freq"]
        self.ctrl_dt = 1.0 / self.ctrl_freq
        self.start_pos = params["start_pos"]
        self.stop_pos = params["stop_pos"][0:3]
        self.accuracy = params["accuracy"]
        self.max_recursion_num = int(params["max_recursion_num"])
        self.uav_radius = params["uav_radius"]
        self.path_insert_point_dist_min = params["path_insert_point_dist_min"]
        self.traj_max_vel = params["traj_max_vel"]
        self.traj_gamma = params["traj_gamma"]

        self.map = map
        self.map.obstacle_dilation(dist=2*self.uav_radius)

        self._find_init_path()
        self._path_correct()
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
        self.init_path = [self.start_pos, self.stop_pos]

    def _path_correct(self):
        waypoints = np.array(self.init_path)

        def _find_viable_point(intp1, intp2, dir_num=32):
            world_vec = (intp2-intp1) / np.linalg.norm(intp2-intp1)
            if world_vec[0] == 0 and world_vec[2] == 0:
                roll = math.pi/2 if world_vec[1] == -1 else -math.pi/2
                pitch = 0.0
            else:
                pitch = math.atan2(world_vec[0], world_vec[2])
                cos_roll = world_vec[0]/math.sin(pitch)
                roll = math.atan2(-world_vec[1], cos_roll)
            
            rot_M = np.array([[math.cos(pitch), math.sin(pitch)*math.sin(roll), math.sin(pitch)*math.cos(roll)],
                            [0, math.cos(roll), -math.sin(roll)],
                            [-math.sin(pitch), math.cos(pitch)*math.sin(roll), math.cos(pitch)*math.cos(roll)]])
            
            angles = np.linspace(0.0, 2*math.pi, num=dir_num, endpoint=False)
            directions = np.array([[math.cos(a), math.sin(a), 0] for a in angles])
            mid_point = (intp1 + intp2) / 2
            trans_directions = np.array([rot_M.dot(dir.T) for dir in directions])

            step_len = self.accuracy
            max_step_num = 1000*dir_num
            step_count = 0
            while step_count < max_step_num:
                fit_dist = np.inf
                best_point = None
                for d in trans_directions:
                    temp_point = mid_point + step_len*d
                    dist_to_l = point2line_dist(intp1, intp2, temp_point)
                    if not self._if_pos_collide_with_obstacles(temp_point) and dist_to_l < fit_dist:
                        fit_dist = dist_to_l
                        best_point = temp_point + self.accuracy*d
                if best_point is not None:
                    return best_point
                step_count = step_count + 1
                step_len = step_len + step_count*self.accuracy
            return None

        def _intersect(p1, p2, count=0):
            dist = np.linalg.norm(p2-p1)
            if dist < self.path_insert_point_dist_min or count >= self.max_recursion_num:
                return np.concatenate(([p1], [p2]))
            seg_num = int(dist/self.accuracy)
            seg_points = np.linspace(p1, p2, num=seg_num, endpoint=True)
            flag = False
            intersect_p1 = None
            intersect_p2 = None
            index = 0

            for p in seg_points:
                index = index + 1
                if self._if_pos_collide_with_obstacles(p):
                    intersect_p1 = p
                    flag = True
                    break
                
            if flag is True:
                for p in seg_points[index:]:
                    if not self._if_pos_collide_with_obstacles(p):
                        intersect_p2 = p
                        break
                
                mid_point = _find_viable_point(intersect_p1, intersect_p2)
                front_way = _intersect(p1, mid_point, count+1)
                after_way = _intersect(mid_point, p2, count+1)
                return np.concatenate((front_way, after_way[1:]))
            else:
                return np.concatenate(([p1], [p2]))

        path = [waypoints[0]]
        for i in range(len(waypoints)-1):
            start_p = waypoints[i]
            end_p = waypoints[i+1]
            segment = _intersect(start_p, end_p)
            path = np.concatenate((path, segment[1:]))

        self.raw_path = path
        self.path = path

        print("\033[0;32;40mInitial feasible path found!\033[0m")

    def _trajectory_plan(self):
        waypoints = self.path
        
        print("\033[7mConducting trajectory planning...\033[0m")

        generator = trajGenerator(np.array(waypoints), max_vel=self.traj_max_vel, gamma=self.traj_gamma)
        self.traj_generator = generator

        timestamp = np.ones(int(self.ctrl_freq*self.episode_len_sec)) * self.ctrl_dt
        timestamp[0] = 0.0
        timestamp = np.cumsum(timestamp)
        self.timestamp = timestamp
        self.pos_trajectory = np.array([generator.get_des_state(t).pos for t in timestamp])

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
