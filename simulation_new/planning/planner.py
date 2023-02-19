import os
import numpy as np
import pickle
from datetime import datetime
from collections import deque
from simulation_new.utils.simulation_utils import draw_trajectory
from simulation_new.utils.map_generate.occupancy_grid_map.general_utils.gridmap import *
from simulation_new.utils.map_generate.occupancy_grid_map.general_utils.add_obstacles import add_obstacles_to_gridmap
from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger

from .competition_planning.traj_gen_1 import TrajGenerator1
from .path_planning_and_minimum_snap.traj_gen_2 import TrajGenerator2

from simulation_new.medium_layer.resampler import Resampler

os.environ['KMP_DUPLICATE_LIB_OK']='True'
folder_name = 'generated_trajectories'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

experiment_settings = {
    'load_trajectory': True,
    'trajectory_filename': 'obs_env_02.19.2023_21.02.56',
    # whether load the existing trajectory or not
    'planning_algorithm': 'path_planning_and_minimum_snap',
    # options: 'competition_planning', 'path_planning_and_minimum_snap'
    'controlling_algorithm': 'pid', # not used
    # options: 'pid'
    'resampling_algorithm': 'adaptive', # not used
    # options: 'adaptive', 'optimization'
}

class Planner():
    """Controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 buffer_size: int = 100,
                 verbose: bool = True,
                 logger: Logger = None
                 ):
        """Initialization of the controller.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', etc.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.EPISODE_LEN_SEC = initial_info["episode_len_sec"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size
        self.OBSTACLES = initial_info["nominal_obstacles_info"]
        self.logger = logger

        self.last_simtime = 0.

        init_pos = [initial_obs[0], initial_obs[2], initial_obs[4]]
        
        # Reset counters and buffers.
        self.reset()

        if experiment_settings['load_trajectory']:
            file_path = folder_name + '/' + experiment_settings['trajectory_filename'] + '.pickle'
            with open(file_path, 'rb') as in_file:
                self.trajectory = pickle.load(in_file)
                path = pickle.load(in_file)
                pos_trajectory = pickle.load(in_file)
                in_file.close()
        else:
            blank_map = BlankGridMap(initial_info["site_size"][0], initial_info["site_size"][1], initial_info["site_size"][2])
            add_obstacles_to_gridmap(blank_map, self.OBSTACLES)
            # blank_map.plot(grid=True)

            start_pos = init_pos
            goal_pos = [initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]]
            
            if experiment_settings['planning_algorithm'] == 'competition_planning':
                # Planning method used in competition 2022
                traj_plan_params = {"ctrl_time": self.EPISODE_LEN_SEC, "ctrl_freq": self.CTRL_FREQ,
                                    "start_pos": start_pos, "stop_pos": goal_pos, "max_recursion_num": 3,
                                    "uav_radius": 0.075, "accuracy": blank_map.cell_size,
                                    "path_insert_point_dist_min": 0.1,"traj_max_vel": 0.5, "traj_gamma": 50}
                planner = TrajGenerator1(blank_map, traj_plan_params)
            
            elif experiment_settings['planning_algorithm'] == 'path_planning_and_minimum_snap':
                # Planning method using a_star, rrt_relevant
                traj_plan_params = {"ctrl_time": self.EPISODE_LEN_SEC, "ctrl_freq": self.CTRL_FREQ,
                                    "start_pos": start_pos, "stop_pos": goal_pos,
                                    "uav_radius": 0.075, "accuracy": blank_map.cell_size,
                                    "path_insert_point_dist_min": 0.1,"traj_max_vel": 10., "traj_gamma": 300000.}
                planner = TrajGenerator2(blank_map, traj_plan_params)
            
            else:
                print('Invalid planning algorithm!')
                exit(-1)
            
            for _ in range(3):
                flag = planner.trajectory_replan()
                if flag:
                    print("\033[4;33;40mMore attempts on trajectory planning may be needed.\033[0m")
                else:
                    print("\033[0;37;42mReplanning Done!\033[0m")
                    break
            self.trajectory = planner.trajectory
            path = planner.path
            pos_trajectory = planner.pos_trajectory
            with open(os.path.join(folder_name, 'obs_env_'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".pickle"), 'wb') as out_file:
                pickle.dump(self.trajectory, out_file)
                pickle.dump(path, out_file)
                pickle.dump(pos_trajectory, out_file)
                out_file.close()

        self.command_time = 0.
        self.resampler = Resampler(self.CTRL_TIMESTEP, "adapt")

        self.rate_buffer = np.array([]) # store the relative time flow rate
        self.s_error_buffer = []
        self.l_error_buffer = []

        # Temporary log for output
        self.max_pln_vel = 0.
        self.max_cmd_vel = 0.

        # Draw the trajectory on PyBullet's GUI.
        self.waypoints = np.array(path)
        self.path_for_plot = pos_trajectory
        self.interEpisodeReset()
        self.plot_path()

    def plot_path(self):
        draw_trajectory(self.waypoints, self.path_for_plot[:,0], self.path_for_plot[:,1], self.path_for_plot[:,2])

    def cmdSimulation(self,
                    sim_time,
                    obs,
                    resample: bool=True,
                    info=None
                    ):
        """Software-only quadrotor commander.

        Args:
            sim_time (float): Outside simualtion time, in seconds.
            obs (ndarray): The quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        dt = sim_time - self.last_simtime
        self.last_simtime = sim_time
        if resample:
            rate, s_error, l_error = self.resampler.get_time_flow_ratio(self.command_time,
                                                    obs,
                                                    self.trajectory)
            self.s_error_buffer.append(s_error)
            self.l_error_buffer.append(l_error)
        else:
            rate = 1.
            # curr_pos = np.array([obs[0], obs[2], obs[4]])
            # ref_s = self.trajectory.eval_length(self.command_time)
            # _, s_near, pos_near = self.trajectory.closest_point_from_cartesian(curr_pos, ref_time=self.command_time)
            # s_error = ref_s - s_near
            # l_error = np.linalg.norm(curr_pos - pos_near)
            # self.s_error_buffer.append(s_error)
            # self.l_error_buffer.append(l_error)

        if self.command_time >= self.trajectory.duration:
            self.command_time = self.trajectory.duration
        else:
            self.command_time += dt * rate
        ref = self.trajectory.eval(self.command_time)
        target_p = ref.pos
        target_v = ref.vel * rate
        target_a = ref.acc * rate**2
        target_omega = ref.omega * rate
        target_yaw = ref.yaw

        if np.linalg.norm(ref.vel) > self.max_pln_vel:
            self.max_pln_vel = np.linalg.norm(ref.vel)
        if np.linalg.norm(target_v) > self.max_cmd_vel:
            self.max_cmd_vel = np.linalg.norm(target_v)
    
        if self.logger != None:
            self.logger.log_extend('calc_timeflow_ratio_time', sim_time)
            self.logger.log_extend('resampled_timestamps', self.command_time)
            self.logger.log_extend('timeflow_ratio', rate)

        return target_p, target_v, target_a, target_omega, target_yaw
    
    def _startup_traj(self, obs, height=1., duration=2.):
        """Software quadrotor start-up hovering commander.
            Call before cmdSimulation.

        Args:
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            height (float): start-up height, in meters.
            duration (float): time duration of the start-up process.

        Returns:
            ndarray: target position trajectory.
            ndarray: target velocity trajectory.
        """
        start_x, start_y, start_z = obs[0], obs[2], obs[4]
        z_error = height - start_z
        step_num = int(round(duration / self.CTRL_TIMESTEP))
        timecount = np.linspace(0, 1, step_num)
        target_p = [[start_x, start_y, start_z + z_error*(1000*t/(1+1000*t))] for t in timecount]
        target_v = [[0, 0, z_error*(1000/(1+1000*t)**2)] for t in timecount]
        return np.array(target_p), np.array(target_v)

    
    def _goto_traj(self, start, dest, duration=3.):
        """Software quadrotor goto commander.
            Call before cmdSimulation.

        Args:
            start (list): The quadrotor's current postion [x y z].
            dest (list): destination coordinates [x y z].
            duration (float): time duration of the goto process.

        Returns:
            ndarray: target position trajectory.
            ndarray: target velocity trajectory.
        """
        start_x, start_y, start_z = start[0], start[1], start[2]
        goal_x, goal_y, goal_z = dest[0], dest[1], dest[2]
        x_error = goal_x - start_x
        y_error = goal_y - start_y
        z_error = goal_z - start_z
        step_num = int(round(duration / self.CTRL_TIMESTEP))
        timecount = np.linspace(0, 1, step_num)
        target_p = [[start_x + x_error*t,
                    start_y + y_error*t, 
                    start_z + z_error*t] for t in timecount]
        target_v = [[x_error / duration,
                    y_error / duration, 
                    z_error / duration] for _ in timecount]
        return np.array(target_p), np.array(target_v)


    def onlineAnalysis(self,
                    time,
                    action,
                    obs,
                    cost,
                    done,
                    info):
        """Conducting online flight analysis, mainly calculate RMSE of tracking performance

        Args:
            time (float): Current simulation time.
            action (List): Most recent applied action.
            obs (List): Most recent observation of the quadrotor state.
            cost (float): Most recent cost.
            done (bool): Most recent done flag.
            info (dict): Most recent information dictionary.

        """
        self.interstep_counter += 1

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.cost_buffer.append(cost)
        self.done_buffer.append(done)
        self.info_buffer.append(info)


    def EpisodeAnalysis(self):
        """Analysis and save episode results

        """
        self.interepisode_counter += 1

        print("Max Velocity Planned: ", self.max_pln_vel)
        print("Max Velocity Commanded: ", self.max_cmd_vel)

        if self.logger != None:
            self.logger.log_extend('startup_time', self.start_up_time)
            self.logger.log_extend('trajectory', self.trajectory)
            self.logger.log_extend('s_error', self.s_error_buffer)
            self.logger.log_extend('l_error', self.l_error_buffer)

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.cost_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0
        self.command_time = 0.

        self.rate_buffer = np.array([])
        self.s_error_buffer = []
        self.l_error_buffer = []

        # Temporary log for output
        self.max_pln_vel = 0.
        self.max_cmd_vel = 0.
        self.max_pln_acc = 0.
        self.max_cmd_acc = 0.

    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
        self.command_time = 0.

        self.rate_buffer = np.array([])
        self.s_error_buffer = []
        self.l_error_buffer = []

        # Temporary log for output
        self.max_pln_vel = 0.
        self.max_cmd_vel = 0.
        self.max_pln_acc = 0.
        self.max_cmd_acc = 0.

#        self.plot_path()