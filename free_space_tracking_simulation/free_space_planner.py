import numpy as np
from collections import deque
from simulation.utils.simulation_utils import draw_trajectory
from simulation.planning.path_planning_and_minimum_snap.TrajGen.trajGen import trajGenerator
from simulation.utils.uav_trajectory.frenet import FrenetTrajectory
from simulation.medium_layer.resampler import Resampler

from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger

# intermediate_waypoints = [
#     [0.2, 1.1, 1.],
#     [2.2, 3.9, 1.3],
#     [4.3, 4., 2.],
#     [4.8, 2.3, 2.1],
#     [2.5, 2.8, 2.4],
#     [1.4, 3.2, 2.1],
#     [0.5, 2.2, 1.8],
#     [1.7, 1., 1.3],
#     [3.7, 0.9, 1.4],
#     [6.1, 2.4, 1.6]
# ]
intermediate_waypoints = [
    [0.5, 0.8, 1.3],
    [2., 2., 1.5],
    [3.5, 3., 1.7]
]
# intermediate_waypoints = [
#     [0.8, 2.1, 1.3],
#     [1.5, 3.7, 1.5],
#     [2.6, 1.9, 1.7],
#     [3.2, 1.4, 1.6],
#     [5., 2., 1.4]
# ]
max_vel = 1.
gamma = 0.5

class Free_Space_Planner():
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
        self.PYB_CLIENT = initial_info["pyb_client"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size
        self.KF = initial_info["quadrotor_kf"]
        self.logger = logger

        init_pos = [initial_obs[0], initial_obs[2], initial_obs[4]]
        # Planning settings
        start_height = 1.
        start_x = 0.
        start_y = 0.
        startup_hovering_duration = 2.

        self.start_traj_p, self.start_traj_v = self._startup_traj(initial_obs, start_height, startup_hovering_duration)
        if (init_pos[0]-start_x)**2 + (init_pos[1]-start_y)**2 >= 0.05:
            goto_traj_p, goto_traj_v = self._goto_traj([init_pos[0], init_pos[1], start_height],
                                        [start_x, start_y, start_height])
            self.start_traj_p = np.concatenate((self.start_traj_p, goto_traj_p), axis=0)
            self.start_traj_v = np.concatenate((self.start_traj_v, goto_traj_v), axis=0)
        self.start_up_time = self.start_traj_p.shape[0]*self.CTRL_TIMESTEP
        
        # Reset counters and buffers.
        self.reset()

        start_pos = [start_x, start_y, start_height]
        goal_pos = [initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]]
        
        waypoints = np.concatenate(([start_pos], intermediate_waypoints, [goal_pos]), axis=0)
        minimum_snap = trajGenerator(np.array(waypoints), max_vel=max_vel, gamma=gamma)
        coef_mat = minimum_snap.get_coef_mat()
        self.trajectory = FrenetTrajectory(coef_mat=coef_mat, order=minimum_snap.order)

        timestamp = np.ones(int(initial_info['ctrl_freq']*initial_info['episode_len_sec'])) * (1. / initial_info['ctrl_freq'])
        timestamp[0] = 0.0
        timestamp = np.cumsum(timestamp)
        self.timestamp = timestamp[timestamp<=minimum_snap.TS[-1]]
        pos_trajectory = np.array([minimum_snap.get_des_state(t).pos for t in timestamp])

        self.command_time = 0.
        self.resampler = Resampler(self.CTRL_TIMESTEP, "adapt")

        self.rate_buffer = np.array([]) # store the relative time flow rate
        self.s_error_buffer = []
        self.l_error_buffer = []

        # Temporary log for output
        self.max_pln_vel = 0.
        self.max_cmd_vel = 0.

        # Draw the trajectory on PyBullet's GUI.
        self.waypoints = waypoints
        self.path_for_plot = pos_trajectory
        self.interEpisodeReset()
    
    def plot_path(self):
        draw_trajectory(self.waypoints, self.path_for_plot[:,0], self.path_for_plot[:,1], self.path_for_plot[:,2], self.PYB_CLIENT)

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

        if sim_time < self.start_up_time:
            iteration = int(sim_time*self.CTRL_FREQ)
            rate = 1.
            # self.rate_buffer = np.append(self.rate_buffer, rate)

            target_p = self.start_traj_p[iteration].copy()
            target_v = self.start_traj_v[iteration].copy()
            target_a = np.zeros(3)
            target_omega = np.zeros(3)
            target_yaw = 0.

        else:
            if resample:
                rate, s_error, l_error = self.resampler.get_time_flow_ratio(self.command_time,
                                                        obs,
                                                        self.trajectory)
                self.s_error_buffer.append(s_error)
                self.l_error_buffer.append(l_error)
            else:
                rate = 1.
                curr_pos = np.array([obs[0], obs[2], obs[4]])
                ref_s = self.trajectory.eval_length(self.command_time)
                _, s_near, pos_near = self.trajectory.closest_point_from_cartesian(curr_pos, ref_time=self.command_time)
                s_error = ref_s - s_near
                l_error = np.linalg.norm(curr_pos - pos_near)
                self.s_error_buffer.append(s_error)
                self.l_error_buffer.append(l_error)

            # self.rate_buffer = np.append(self.rate_buffer, rate)
            if self.command_time >= self.trajectory.duration:
                self.command_time = self.trajectory.duration
            else:
                self.command_time += self.CTRL_TIMESTEP * rate
            ref = self.trajectory.eval(self.command_time)
            target_p = ref.pos
            target_v = ref.vel * rate

            # if self.rate_buffer.size == 1:
            #     rate_dot = 0
            # elif self.rate_buffer.size == 2:
            #     rate_dot = (self.rate_buffer[-1] - self.rate_buffer[-2]) / self.CTRL_TIMESTEP
            # else:
            #     rate_dot = (3*self.rate_buffer[-1] - 4*self.rate_buffer[-2] + self.rate_buffer[-3]) / (2*self.CTRL_TIMESTEP)

            target_a = ref.acc * rate**2
            # target_a = ref.acc * rate**2 + rate_dot * ref.vel
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

        self.plot_path()