#!~/anaconda3/envs/teb/bin/python
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
from uav_trajectory.frenet import FrenetTrajectory

def coef_of_difficulty(curv, tors, vel, k_curv=3., k_tors=5.):
    """
    Calculate the coefficient of difficulty of tracking the current trajectory,
    which will be used to calculate the relative time flowing rate.

    Args:
        curv: (float) current curvature
        tors: (float) current torsion
        vel: (1*3 ndarray) the original planned velocity of current trajectory position

    Returns:
        (float) coef_of_difficulty in range [0,inf)

    Raises:
        None
    """
    return np.linalg.norm(vel) * (k_curv*curv + k_tors*tors)

def Gaussian_fcn(x, mean=0, amp=1.4, sigma=3, min=0.1):
    if not isinstance(x, list):
        return amp * np.exp(-(x-mean)**2 / (2*sigma**2)) + min
    else:
        result = []
        for t in x:
            result.append(amp * np.exp(-(t-mean)**2 / (2*sigma**2)) + min)
        return result


class FrenetTrackingTrajectory(FrenetTrajectory):

    def __init__(self, filename: str):
        super().__init__(filename)

    def get_flow_rate(self, t):
        curv = self.eval_curvature([t])[0]
        tors = self.eval_torsion([t])[0]
        des_states = self.eval(t)
        d_vel = des_states.vel
        difficulty_coef = coef_of_difficulty(curv, tors, d_vel)
        return Gaussian_fcn(difficulty_coef)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, num=5000, endpoint=True)
    y = Gaussian_fcn(x)
    fig1 = plt.figure('Gaussian Function')
    ax1 = fig1.add_axes([0.05,0.05,0.95,0.95])
    ax1.plot(x, y)
    
    tracktraj = FrenetTrackingTrajectory(os.path.abspath(os.path.dirname(__file__)) +
                                        '/generated_trajectory/test.csv')
    time = np.linspace(0, tracktraj.duration, num=5000, endpoint=True)
    rates = []
    for t in time: rates.append(tracktraj.get_flow_rate(t))
    fig2 = plt.figure('flow rate along trajectory')
    ax2 = fig2.add_axes([0.05,0.05,0.95,0.95])
    ax2.plot(time, rates)

    plt.show()