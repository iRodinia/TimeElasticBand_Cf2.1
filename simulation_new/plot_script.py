import os
import numpy as np
import matplotlib.pyplot as plt

from simulation.utils.map_generate.occupancy_grid_map.general_utils.gridmap import *


def plot_tracking_status(resampled_log, direct_log):
    """draw 4 subplots:
        1. 3D reference trajectory *2 and real trajectory *2
        2. x-axis original reference, direct tracking positions, resampled reference and resampled tracking positions
        3. y-axis original reference, direct tracking positions, resampled reference and resampled tracking positions
        4. z-axis original reference, direct tracking positions, resampled reference and resampled tracking positions

        original reference: green, 
        direct tracking states: blue, 
        resampled reference: orange, 
        resampled tracking states: red

    Args:
        TBD
    """
    timestamps1 = resampled_log['timestamps'][0]
    states1 = resampled_log['states'][0]
    references1 = resampled_log['targets'][0]

    timestamps2 = direct_log['timestamps'][0] - timestamps1[-1]
    states2 = direct_log['states'][0]
    references2 = direct_log['targets'][0]

    original_refs = references2
    resampled_refs = references1
    direct_states = states2
    resampled_states = states1


    fig_tracking = plt.figure("Position Tracking Status", figsize=(13, 3.2))
    ax1 = fig_tracking.add_subplot(141, projection='3d')
    ax2 = fig_tracking.add_subplot(142)
    ax3 = fig_tracking.add_subplot(143)
    ax4 = fig_tracking.add_subplot(144)

    ax1.plot(original_refs[0,:], original_refs[1,:], original_refs[2,:], color='green')
    ax1.plot(direct_states[0,:], direct_states[1,:], direct_states[2,:], color='blue')
    ax1.plot(resampled_states[0,:], resampled_states[1,:], resampled_states[2,:], color='red')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    ax2.plot(timestamps2, original_refs[0,:], color='green')
    ax2.plot(timestamps1, resampled_refs[0,:], color='orange')
    ax2.plot(timestamps2, direct_states[0,:], color='blue')
    ax2.plot(timestamps1, resampled_states[0,:], color='red')
    ax2.set_xlabel('time /s')
    ax2.set_ylabel('x /m')

    ax3.plot(timestamps2, original_refs[1,:], color='green')
    ax3.plot(timestamps1, resampled_refs[1,:], color='orange')
    ax3.plot(timestamps2, direct_states[1,:], color='blue')
    ax3.plot(timestamps1, resampled_states[1,:], color='red')
    ax3.set_xlabel('time /s')
    ax3.set_ylabel('y /m')

    ax4.plot(timestamps2, original_refs[2,:], color='green')
    ax4.plot(timestamps1, resampled_refs[2,:], color='orange')
    ax4.plot(timestamps2, direct_states[2,:], color='blue')
    ax4.plot(timestamps1, resampled_states[2,:], color='red')
    ax4.set_xlabel('time /s')
    ax4.set_ylabel('z /m')

    fig_tracking.subplots_adjust(
        left=0,
        bottom=0.143,
        right=0.971,
        top=0.948,
        wspace=0.224,
        hspace=0.2
    )



def plot_trajectory_tracking_error_and_TRMSE(resampled_log, direct_log):
    """draw 1 plot:
        1. p(t)-p^(t)

        direct tracking error: blue, 
        resampled tracking error: red

    Args:
        TBD
    """
    timestamps1 = resampled_log['timestamps'][0]
    data_dict1 = resampled_log['other_data'].item()
    states1 = resampled_log['states'][0]
    references1 = resampled_log['targets'][0]
    startup_time1 = data_dict1['startup_time'][0]
    cmd_timestamps1 = timestamps1[timestamps1 >= startup_time1]
    traj_len1 = cmd_timestamps1.shape[0]

    timestamps2 = direct_log['timestamps'][0] - timestamps1[-1]
    data_dict2 = direct_log['other_data'].item()
    states2 = direct_log['states'][0]
    references2 = direct_log['targets'][0]
    startup_time2 = data_dict2['startup_time'][0]
    cmd_timestamps2 = timestamps2[timestamps2 >= startup_time2]
    traj_len2 = cmd_timestamps2.shape[0]

    tracking_error1 = []
    tmse1 = 0.
    for i in range(timestamps1.shape[0]):
        e = ((states1[0,i]-references1[0,i])**2 + (states1[1,i]-references1[1,i])**2 + (states1[2,i]-references1[2,i])**2)**0.5
        tracking_error1.append(e)
        tmse1 += e**2
    trmse1 = np.sqrt(tmse1 / timestamps1.shape[0])
    
    tracking_error2 = []
    tmse2 = 0.
    for i in range(timestamps2.shape[0]):
        e = ((states2[0,i]-references2[0,i])**2 + (states2[1,i]-references2[1,i])**2 + (states2[2,i]-references2[2,i])**2)**0.5
        tracking_error2.append(e)
        tmse2 += e**2
    trmse2 = np.sqrt(tmse2 / timestamps2.shape[0])

    fig_trmse = plt.figure("Trajectory Tracking Error", figsize=(5,3.2))
    ax = fig_trmse.add_axes([0.15,0.15,0.8,0.8])
    ax.plot(timestamps1[-traj_len1:], tracking_error1[-traj_len1:], color='red')
    ax.plot(timestamps2[-traj_len2:], tracking_error2[-traj_len2:], color='blue')
    ax.set_xlabel('time /s')
    ax.set_ylabel(r'$\Vert r-\hat{r} \Vert_2$ /m')

    print("Direct Trajectory tracking RMSE: ", trmse2)
    print("Resampled Trajectory tracking RMSE: ", trmse1)



def plot_frenet_tracking_error_and_PRMSE(resampled_log, direct_log):
    """draw 2 subplots:
        1. s error between the reference and tracking positions
        2. l error between the reference and tracking positions

        direct tracking error: blue, 
        resampled tracking error: red

    Args:
        TBD
    """
    timestamps1 = resampled_log['timestamps'][0]
    data_dict1 = resampled_log['other_data'].item()
    startup_time1 = data_dict1['startup_time'][0]
    s_error1 = data_dict1['s_error'][0]
    l_error1 = data_dict1['l_error'][0]
    cmd_timestamps1 = timestamps1[timestamps1 >= startup_time1]

    timestamps2 = direct_log['timestamps'][0] - timestamps1[-1]
    data_dict2 = direct_log['other_data'].item()
    startup_time2 = data_dict2['startup_time'][0]
    s_error2 = data_dict2['s_error'][0]
    l_error2 = data_dict2['l_error'][0]
    cmd_timestamps2 = timestamps2[timestamps2 >= startup_time2]

    pmse1 = 0.
    for i in range(cmd_timestamps1.shape[0]):
        pmse1 += l_error1[i]**2
    prmse1 = np.sqrt(pmse1 / cmd_timestamps1.shape[0])
    
    pmse2 = 0.
    for i in range(cmd_timestamps2.shape[0]-1):
        pmse2 += l_error2[i]**2
    prmse2 = np.sqrt(pmse2 / cmd_timestamps2.shape[0])

    fig_frenet_error = plt.figure("Frenet Tracking Error", figsize=(5,3.2))
    ax1 = fig_frenet_error.add_subplot(121)
    ax2 = fig_frenet_error.add_subplot(122)

    ax1.plot(cmd_timestamps1, s_error1, color='red')
    ax1.plot(cmd_timestamps2[:-1], s_error2, color='blue')
    ax1.set_xlabel('time /s')
    ax1.set_ylabel(r'$s_e$ /m')

    ax2.plot(cmd_timestamps1, l_error1, color='red')
    ax2.plot(cmd_timestamps2[:-1], l_error2, color='blue')
    ax2.set_xlabel('time /s')
    ax2.set_ylabel(r'$\sqrt{l_e^2+d_e^2}$ /m')

    fig_frenet_error.subplots_adjust(
        left=0.145,
        bottom=0.155,
        right=0.971,
        top=0.971,
        wspace=0.479,
        hspace=0.2
    )

    print("Direct Positional track RMSE: ", prmse2)
    print("Resampled Positional track RMSE: ", prmse1)


def plot_resampled_timeflow_rate(logs):
    data_dict = logs['other_data'].item()
    timestamps = logs['timestamps'][0]
    resampled_timestamps = data_dict['resampled_timestamps']
    startup_time = data_dict['startup_time'][0]
    cmd_timestamps = timestamps[timestamps >= startup_time]
    rate = data_dict['timeflow_ratio']

    fig_timeflow = plt.figure("Time Flow Rate", figsize=(5,3.2))
    ax = fig_timeflow.add_axes([0.15,0.15,0.75,0.75])
    ax.plot(cmd_timestamps, rate)
    ax.set_xlabel('time /s')
    ax.set_ylabel('c(t)')


def plot_trajecotry_status(logs):
    data_dict = logs['other_data'].item()
    trajectory = data_dict['trajectory'][0]
    trajectory.print_traj_status()


def calculate_tracking_params(logs, with_resampler=True):
    timestamps = logs['timestamps'][0]
    data_dict = logs['other_data'].item()
    startup_time = data_dict['startup_time'][0]
    trajectory = data_dict['trajectory'][0]
    cmd_timestamps = timestamps[timestamps >= startup_time]

    states = logs['states'][0]
    references = logs['targets'][0]
    resampled_timestamps = data_dict['resampled_timestamps']

    if with_resampler:
    #########################################################################################################
        original_vel_refs = np.array([trajectory.eval(t-startup_time).vel for t in cmd_timestamps])
        original_vels = np.array([np.linalg.norm(v) for v in original_vel_refs])
        max_original_vel = original_vels.max()
        print("Original Planned Max Velocity: ", max_original_vel)

    #########################################################################################################
    traj_len = cmd_timestamps.shape[0]
    real_vel_refs = references[3:6,-traj_len:].T
    real_vels = np.array([np.linalg.norm(v) for v in real_vel_refs])
    max_real_vel = real_vels.max()
    print("Max Velocity Commanded: ", max_real_vel)

    #########################################################################################################



if __name__ == '__main__':
    
    with_resampler_log_file_name = "with_resampler-02.10.2023_20.17.24.npy"
    no_resampler_log_file_name = "no_resampler-02.10.2023_20.18.41.npy"

    log_path1 = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/results/' + with_resampler_log_file_name
    log_path2 = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/results/' + no_resampler_log_file_name

    logs1 = np.load(log_path1, allow_pickle=True)
    logs2 = np.load(log_path2, allow_pickle=True)

    plot_tracking_status(logs1, logs2)

    plot_trajectory_tracking_error_and_TRMSE(logs1, logs2)
    plot_frenet_tracking_error_and_PRMSE(logs1, logs2)

    plot_resampled_timeflow_rate(logs1)

    # calculate_tracking_params(logs1, with_resampler=True)
    # calculate_tracking_params(logs2, with_resampler=False)

    plot_trajecotry_status(logs1)

    plt.show()