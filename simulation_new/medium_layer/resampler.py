import numpy as np
from simulation_new.utils.uav_trajectory.frenet import FrenetTrajectory
from .adaptive_sampling.adaptive_sampler import AdaptiveSampler


class Resampler:
    def __init__(self, dt, type: str):
        self.type = type
        self.dt = dt
        self.time_ahead = 0.15
        if type == "adapt":
            pid_sampler_params = {
                'max_acc_pred': 1.3,
                'max_acc_curr': 1.1,
                'time_ahead': self.time_ahead,
                'dt': self.dt,
                'torsion_weight': 1.2,  # assume the weight of curvature is 1
                                        # calculate trajectory bending degree
                'l_error_weight': 2., # assume the weight of s_error is 1
                'k_D': 0.1,
                'error_threshold': 0.08
            }
            self.sampler = AdaptiveSampler(pid_sampler_params)
    
    def get_time_flow_ratio(self, time, obs, trajectory: FrenetTrajectory):
        """Calculate time flowing ratio of current reference

        Args:
            time (float): current control time
            obs (ndarray): The state observation of the quadrotor
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            trajectory (FrenetTrajectory): Reference trajectory
        Return:
            ratio
        """
        curr_pos = np.array([obs[0], obs[2], obs[4]])
        curr_vel = np.array([obs[1], obs[3], obs[5]])
        curr_ref_s = trajectory.eval_length(time)
        curr_ref_curv = trajectory.eval_curvature(time)
        curr_ref_tors = trajectory.eval_torsion(time)

        pred_ref_curv = trajectory.eval_curvature(time + self.time_ahead)
        pred_ref_tors = trajectory.eval_torsion(time + self.time_ahead)

        _, curr_s_near, curr_pos_near = trajectory.closest_point_from_cartesian(curr_pos, ref_time=time)
        curr_s_error = curr_ref_s - curr_s_near
        curr_l_error = np.linalg.norm(curr_pos - curr_pos_near)

        if self.type == "adapt":
            self.sampler.refresh_error(time, [curr_s_error, curr_l_error], curr_ref_curv, curr_ref_tors, pred_ref_curv, pred_ref_tors)
        else:
            print("Invalid sampler type!")
            exit(-1)

        if self.type == "adapt":
            ratio = self.sampler.get_ratio_output()
        else:
            print("Invalid sampler type!")
            exit(-1)
        
        return ratio, curr_s_error, curr_l_error
