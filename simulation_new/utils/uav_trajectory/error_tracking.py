#!~/anaconda3/envs/teb/bin/python

import numpy as np
import uav_trajectory.trajectory as traj

class ErrorTrackingTrajectory(traj.Trajectory):

    def __init__(self, filename: str):
        super().__init__()
        self.loadcsv(filename)
        self.derivative1 = [poly.derivative() for poly in self.polynomials]