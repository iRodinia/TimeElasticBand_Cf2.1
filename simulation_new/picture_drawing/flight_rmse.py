import os
import numpy as np

from simulation_new.utils.map_generate.occupancy_grid_map.general_utils.gridmap import *
from simulation_new.utils.uav_trajectory.frenet import FrenetTrajectory

log_file_name = "save-flight-12.20.2022_20.49.38.npy"
path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + '/results/' + log_file_name
logs = np.load(path, allow_pickle=True)

timestamps = logs['timestamps'][0]
states=logs['states'][0]
references=logs['targets'][0]
controls=logs['controls'][0]
data_dict=logs['other_data'].item()
startup_time = data_dict['startup_time'][0]
trajectory = data_dict['trajectory'][0]

ctrl_dt = timestamps[-1] - timestamps[-2]

rate_t = data_dict['calc_timeflow_ratio_time']
rate = data_dict['timeflow_ratio']

eval_tracking_t = []
eval_tracking_pos = []
MSE = 0.
for i in range(timestamps.shape[0]):
    t = timestamps[i]
    pos = states[0:3,i]
    if t > startup_time:
        cmd_time = t - startup_time
        eval_tracking_t.append(t)
        _, _, near_pos = trajectory.closest_point_from_cartesian(pos, ref_time=cmd_time, accuracy=0.01)
        eval_tracking_pos.append(near_pos)
        MSE += ((pos[0]-near_pos[0])**2 + (pos[1]-near_pos[1])**2 + (pos[2]-near_pos[2])**2) * ctrl_dt
MSE = MSE / (timestamps[-1] - timestamps[0])
RMSE = np.sqrt(MSE)
print(RMSE)