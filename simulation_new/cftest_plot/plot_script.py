import os
import numpy as np
import matplotlib.pyplot as plt

def cal_len(curr_pos, tar_pos):
    return np.sqrt((curr_pos[0] - tar_pos[0]) ** 2 + (curr_pos[1] - tar_pos[1]) ** 2 + (curr_pos[2] - tar_pos[2]) ** 2 )

def dot(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]

def vec_len(vec):
    return np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)


def pos_cal(pos, mode): # mode = 1: real -> optitrack
                        # else    : opti -> real
    OFFSET = [0.01, -0.03, 0]
    if mode == 1:
        return [pos[0] + OFFSET[0], pos[1] + OFFSET[1], pos[2] + OFFSET[2]]
    else:
        return [pos[0] - OFFSET[0], pos[1] - OFFSET[1], pos[2] - OFFSET[2]]


def find_nearest_point(index, curr_list, target_list):
    min = np.inf
    min_index = -1
    for i in range(np.shape(target_list)[1]):
        len = cal_len([curr_list[0][index], curr_list[1][index], curr_list[2][index]], 
                      [target_list[0][i], target_list[1][i], target_list[2][i]] )
        if len <= min:
            min_index = i 
            min = len
    return min_index


def acc_nearest_pos(pos_list, target_list, curr_index, target_index):
    if target_index == 0 or target_index == target_list.shape[1]-1:
        return [target_list[0][target_index], 
                target_list[1][target_index], 
                target_list[2][target_index]]

    real_vector = [target_list[0][target_index] - pos_list[0][curr_index], 
                   target_list[1][target_index] - pos_list[1][curr_index], 
                   target_list[2][target_index] - pos_list[2][curr_index]]
    
    vec1 = [target_list[0][target_index] - target_list[0][target_index - 1], 
            target_list[1][target_index] - target_list[1][target_index - 1], 
            target_list[2][target_index] - target_list[2][target_index - 1]]
    
    vec2 = [target_list[0][target_index + 1] - target_list[0][target_index], 
            target_list[1][target_index + 1] - target_list[1][target_index], 
            target_list[2][target_index + 1] - target_list[2][target_index]]
    dot1 = dot(real_vector, vec1)
    dot2 = dot(real_vector, vec2)
    if dot1 >= 0:
        d1 = dot1 / (vec_len(vec1)**2)
        vec1[0] = vec1[0] * (1 - d1)
        vec1[1] = vec1[1] * (1 - d1)
        vec1[2] = vec1[2] * (1 - d1)
        return [vec1[0] + target_list[0][target_index - 1], 
                vec1[1] + target_list[1][target_index - 1], 
                vec1[2] + target_list[2][target_index - 1]]
    elif dot2 <= 0:
        d2 = - dot2 / (vec_len(vec2)**2)
        vec2[0] = vec2[0] * d2
        vec2[1] = vec2[1] * d2
        vec2[2] = vec2[2] * d2
        return [vec2[0] + target_list[0][target_index], 
                vec2[1] + target_list[1][target_index], 
                vec2[2] + target_list[2][target_index]]
    else:
        return [target_list[0][target_index], 
                target_list[1][target_index], 
                target_list[2][target_index]]


def calc_nearest_point_list(real_pos_list, target_pos_list):
    assert real_pos_list.shape[1] <= target_pos_list.shape[1]

    near_pos_list = []
    for i in range(real_pos_list.shape[1]):
        near_ref_idx = find_nearest_point(i, real_pos_list, target_pos_list)
        near_ref_pos = acc_nearest_pos(real_pos_list, target_pos_list, i, near_ref_idx)
        near_pos_list.append(near_ref_pos)
    return np.array(near_pos_list).T


def seperate_pos_from_dict(datadict, dict_type):
    if dict_type == 'states':
        position_x = datadict[0]
        position_y = datadict[2]
        position_z = datadict[4]
        return np.array([position_x, position_y, position_z])
    elif dict_type == 'targets':
        target_x = datadict[0]
        target_y = datadict[1]
        target_z = datadict[2]
        return np.array([target_x, target_y, target_z])
    else:
        print("Wrong type!")
        exit(-1)


def target_modification(ref_arrow):
    for i in range(ref_arrow.shape[1]):
        x = ref_arrow[0,i]
        y = ref_arrow[1,i]
        z = ref_arrow[2,i]
        pos_new = pos_cal([x, y, z], 1)
        ref_arrow[0,i] = pos_new[0]
        ref_arrow[1,i] = pos_new[1]
        ref_arrow[2,i] = pos_new[2] + 0.03
    return ref_arrow

    
obs_pos_1_opti = [1.10, 0.51, 0.45]
obs_pos_2_opti = [2,58, 1.41, 0.45]
obs_pos_3_opti = [2.24, 2.38, 0.45]
obs_pos_4_opti = [1.55, 1.63, 0.45]
obs_pos_5_opti = [0.96, 2.31, 0.45]
obs_pos_6_opti = [0.38, 1.26, 0.32]

obs_dict = {"obs1":{"type":"box", "center":pos_cal(obs_pos_1_opti, 1), 
                    "half_extend":[0.165, 0.165, 0.45], "yaw_rad":0, "fixed":True}, 
            "obs2":{"type":"box", "center":pos_cal(obs_pos_2_opti, 1), 
                    "half_extend":[0.165, 0.165, 0.45], "yaw_rad":0., "fixed":True},
            "obs3":{"type":"box", "center":pos_cal(obs_pos_3_opti, 1), 
                    "half_extend":[0.165, 0.165, 0.45], "yaw_rad":0, "fixed":True},
            "obs4":{"type":"box", "center":pos_cal(obs_pos_4_opti, 1), 
                    "half_extend":[0.165, 0.165, 0.45], "yaw_rad":0, "fixed":True},
            "obs5":{"type":"box", "center":pos_cal(obs_pos_5_opti, 1), 
                    "half_extend":[0.165, 0.165, 0.45], "yaw_rad":0., "fixed":True},
            "obs6":{"type":"box", "center":pos_cal(obs_pos_6_opti, 1), 
                    "half_extend":[0.165, 0.165, 0.165], "yaw_rad":0., "fixed":True}
            
            }
    


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

    timestamps2 = direct_log['timestamps'][0]
    states2 = direct_log['states'][0]
    references2 = direct_log['targets'][0]

    original_refs = target_modification(references2)
    resampled_refs = target_modification(references1)
    direct_states = states2
    resampled_states = states1

    fig_tracking = plt.figure("Position Tracking Status", figsize=(13, 3.2))
    ax1 = fig_tracking.add_subplot(141, projection='3d')
    ax2 = fig_tracking.add_subplot(142)
    ax3 = fig_tracking.add_subplot(143)
    ax4 = fig_tracking.add_subplot(144)

    ax1.plot(resampled_refs[0,:], resampled_refs[1,:], resampled_refs[2,:], color='orange')
    ax1.plot(direct_states[0,:], direct_states[2,:], direct_states[4,:], color='blue')
    ax1.plot(resampled_states[0,:], resampled_states[2,:], resampled_states[4,:], color='red')
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
    ax3.plot(timestamps2, direct_states[2,:], color='blue')
    ax3.plot(timestamps1, resampled_states[2,:], color='red')
    ax3.set_xlabel('time /s')
    ax3.set_ylabel('y /m')

    ax4.plot(timestamps2, original_refs[2,:], color='green')
    ax4.plot(timestamps1, resampled_refs[2,:], color='orange')
    ax4.plot(timestamps2, direct_states[4,:], color='blue')
    ax4.plot(timestamps1, resampled_states[4,:], color='red')
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


def plot_frenet_tracking_error_and_PRMSE(resampled_log, direct_log):
    """draw 1 plot:
        1. p(t)-p^(t)

        direct tracking error: blue, 
        resampled tracking error: red

    Args:
        TBD
    """
    timestamps1 = resampled_log['timestamps'][0]
    states1 = resampled_log['states'][0]
    references1 = resampled_log['targets'][0]

    timestamps2 = direct_log['timestamps'][0]
    states2 = direct_log['states'][0]
    references2 = direct_log['targets'][0]

    original_refs = target_modification(references2)
    resampled_refs = target_modification(references1)

    direct_refp = seperate_pos_from_dict(original_refs, 'targets')
    resampled_refp = seperate_pos_from_dict(resampled_refs, 'targets')
    direct_pos = seperate_pos_from_dict(states2, 'states')
    resampled_pos = seperate_pos_from_dict(states1, 'states')

    resampled_nearest_pos_list = calc_nearest_point_list(resampled_pos, resampled_refp)
    direct_nearest_pos_list = calc_nearest_point_list(direct_pos, direct_refp)

    resampled_error = []
    pmse1 = 0.
    for i in range(timestamps1.shape[0]):
        e = ((resampled_pos[0,i]-resampled_nearest_pos_list[0,i])**2 + 
            (resampled_pos[1,i]-resampled_nearest_pos_list[1,i])**2 + 
            (resampled_pos[2,i]-resampled_nearest_pos_list[2,i])**2)**0.5
        resampled_error.append(e)
        if i < timestamps1.shape[0]-1:
            pmse1 += e**2 * (timestamps1[i+1] - timestamps1[i])
    prmse1 = np.sqrt(pmse1 / timestamps1[-1])
    
    direct_error = []
    pmse2 = 0.
    for i in range(timestamps2.shape[0]):
        e = ((direct_pos[0,i]-direct_nearest_pos_list[0,i])**2 + 
            (direct_pos[1,i]-direct_nearest_pos_list[1,i])**2 + 
            (direct_pos[2,i]-direct_nearest_pos_list[2,i])**2)**0.5
        direct_error.append(e)
        if i < timestamps2.shape[0]-1:
            pmse2 += e**2 * (timestamps2[i+1] - timestamps2[i])
    prmse2 = np.sqrt(pmse2 / timestamps2[-1])

    fig_trmse = plt.figure("Frenet Tracking Error", figsize=(5,3.2))
    ax = fig_trmse.add_axes([0.15,0.15,0.8,0.8])
    ax.plot(timestamps1, resampled_error, color='red')
    ax.plot(timestamps2, direct_error, color='blue')
    ax.set_xlabel('time /s')
    ax.set_ylabel(r'$\Vert r-\hat{r} \Vert$ /m')

    print("Direct PRMSE: ", prmse2)
    print("Resampled PRMSE: ", prmse1)



def plot_pos_tracking_error_and_TRMSE(resampled_log, direct_log):
    """draw 1 plot:
        1. p(t)-p^(t)

        direct tracking error: blue, 
        resampled tracking error: red

    Args:
        TBD
    """
    timestamps1 = resampled_log['timestamps'][0]
    states1 = resampled_log['states'][0]
    references1 = resampled_log['targets'][0]

    timestamps2 = direct_log['timestamps'][0]
    states2 = direct_log['states'][0]
    references2 = direct_log['targets'][0]

    original_refs = target_modification(references2)
    resampled_refs = target_modification(references1)

    direct_refp = seperate_pos_from_dict(original_refs, 'targets')
    resampled_refp = seperate_pos_from_dict(resampled_refs, 'targets')
    direct_pos = seperate_pos_from_dict(states2, 'states')
    resampled_pos = seperate_pos_from_dict(states1, 'states')

    resampled_error = []
    tmse1 = 0.
    for i in range(timestamps1.shape[0]):
        e = ((resampled_pos[0,i]-resampled_refp[0,i])**2 + 
            (resampled_pos[1,i]-resampled_refp[1,i])**2 + 
            (resampled_pos[2,i]-resampled_refp[2,i])**2)**0.5
        resampled_error.append(e)
        if i < timestamps1.shape[0]-1:
            tmse1 += e**2 * (timestamps1[i+1] - timestamps1[i])
    trmse1 = np.sqrt(tmse1 / timestamps1[-1])
    
    direct_error = []
    tmse2 = 0.
    for i in range(timestamps2.shape[0]):
        e = ((direct_pos[0,i]-direct_refp[0,i])**2 + 
            (direct_pos[1,i]-direct_refp[1,i])**2 + 
            (direct_pos[2,i]-direct_refp[2,i])**2)**0.5
        direct_error.append(e)
        if i < timestamps2.shape[0]-1:
            tmse2 += e**2 * (timestamps2[i+1] - timestamps2[i])
    trmse2 = np.sqrt(tmse2 / timestamps2[-1])

    fig_trmse = plt.figure("Trajectory Tracking Error", figsize=(5,3.2))
    ax = fig_trmse.add_axes([0.15,0.15,0.8,0.8])
    ax.plot(timestamps1, resampled_error, color='red')
    ax.plot(timestamps2, direct_error, color='blue')
    ax.set_xlabel('time /s')
    ax.set_ylabel(r'$\Vert r-\hat{r} \Vert$ /m')

    print("Direct PRMSE: ", trmse2)
    print("Resampled PRMSE: ", trmse1)





if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.dirname(__file__))
    complete_file = abs_path + "/complete3" + ".npy"
    fail_file = abs_path + "/fall3" + ".npy"

    complete_data = np.load(complete_file, allow_pickle=True)
    fail_data = np.load(fail_file, allow_pickle=True)
    # only contain: 'states', 'targets', 'timestamps'


    plot_frenet_tracking_error_and_PRMSE(complete_data, fail_data)
    plot_pos_tracking_error_and_TRMSE(complete_data, fail_data)
    plot_tracking_status(complete_data, fail_data)
    plt.show()

