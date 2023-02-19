import numpy as np
from collections import deque

class OptiSampler:
    def __init__(self, sample_dt):
        self.dt = sample_dt
        self.kp = 1.
        self.ki = 0.
        self.kd = 0.
        self.buffer_size = 100
        self.cost_buffer = deque([], maxlen=self.buffer_size)
    
    def _compute_cost(self, target, state, curvature, torsion, cost_to_goal):
        """compute cost of current step

        Args:
            target (ndarray): The target(reference) signal to the quadrotor
                [x, y, z, x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot, p, q, r, yaw]
            state (ndarray): The observation of the quadrotor's current state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
            curvature (float): The current reference trajecotry curvature
            torsion (float): The current reference trajecotry torsion
            cost_to_gaol (float or ndarray): The cost to drive reference signal forward,
                can be set as: time to the end, distance to the end

        """
        pos_error = target[[0,1,2]] - state[[0,2,4]]
        vel_error = target[[3,4,5]] - state[[1,3,5]]
        Q_coef = 1.0 * (2.0 * curvature + 1.0 * torsion)
        Q = Q_coef * np.eye(3)
        R_coef = 2.0

        if isinstance(cost_to_goal, float):
            R = R_coef
            cost = pos_error * Q * pos_error.T + R * cost_to_goal**2
        else:
            R = R_coef * np.eye(len(cost_to_goal))
            cost = pos_error * Q * pos_error.T + cost_to_goal * R * cost_to_goal.T
        
        self.cost_buffer.extend(cost)