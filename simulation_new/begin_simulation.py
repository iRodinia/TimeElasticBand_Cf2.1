"""
Main Simulation Script

Run as:

    $ python3 getting_started.py --overrides ./sim_parameters.yaml

"""
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import math
from rich.tree import Tree
from rich import print

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.utils import sync
from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger
from simulation_new.planning.planner import Planner
from simulation_new.utils.map_generate.environment_construction import constructHeightFieldDictFromImg

from pycrazyswarm import Crazyswarm

def diff(pos, last_pos, dt):
    v_x = (pos[0] - last_pos[0]) / dt
    v_y = (pos[1] - last_pos[1]) / dt
    v_z = (pos[2] - last_pos[2]) / dt
    return [v_x, v_y, v_z]

def quat2eulers(q0:float, q1:float, q2:float, q3:float):
    """
    Compute yaw-pitch-roll Euler angles from a quaternion.
    Args
    ----
        q0: Scalar component of quaternion.
        q1, q2, q3: Vector components of quaternion.
    Returns
    -------
        (roll, pitch, yaw) (tuple): 321 Euler angles in radians
    """
    roll = math.atan2(
        2 * ((q2 * q3) + (q0 * q1)),
        q0**2 - q1**2 - q2**2 + q3**2
    )  # radians
    pitch = math.asin(2 * ((q1 * q3) - (q0 * q2)))
    yaw = math.atan2(
        2 * ((q1 * q2) + (q0 * q3)),
        q0**2 + q1**2 - q2**2 - q3**2
    )
    return [roll, pitch, yaw]

def pos_cal(pos, mode): # mode = 1: plan -> opti
                        # else    : opti -> plan
    OFFSET = [-1.95, -1.89, 0.03]
    if mode == 1:
        return [pos[0] + OFFSET[0], pos[1] + OFFSET[1], pos[2] + OFFSET[2]]
    else:
        return [pos[0] - OFFSET[0], pos[1] - OFFSET[1], pos[2] - OFFSET[2]]
    
obs_pos_1_opti = [-1.27, -0.75, 0.45]
obs_pos_2_opti = [-0.82, 0.42, 0.45]
obs_pos_3_opti = [0.10, 1.22, 0.45]
obs_pos_4_opti = [0.15, 0.18, 0.45]
obs_pos_5_opti = [0.29, -1.06, 0.45]
obs_pos_6_opti = [1.14, 0.17, 0.32]

obs_dict = {"obs1":{"type":"box", "center":pos_cal(obs_pos_1_opti, 0), 
                    "half_extend":[0.165, 0.165, 0.45], "yaw_rad":0, "fixed":True}, 
            "obs2":{"type":"box", "center":pos_cal(obs_pos_2_opti, 0), 
                    "half_extend":[0.165, 0.165, 0.45], "yaw_rad":0., "fixed":True},
            "obs3":{"type":"box", "center":pos_cal(obs_pos_3_opti, 0), 
                    "half_extend":[0.165, 0.165, 0.45], "yaw_rad":0, "fixed":True},
            "obs4":{"type":"box", "center":pos_cal(obs_pos_4_opti, 0), 
                    "half_extend":[0.165, 0.165, 0.45], "yaw_rad":0, "fixed":True},
            "obs5":{"type":"box", "center":pos_cal(obs_pos_5_opti, 0), 
                    "half_extend":[0.165, 0.165, 0.45], "yaw_rad":0., "fixed":True},
            "obs6":{"type":"box", "center":pos_cal(obs_pos_6_opti, 0), 
                    "half_extend":[0.165, 0.165, 0.165], "yaw_rad":0., "fixed":True}
            }

mid_waypts = [
    [-0.77, -0.64, 0.5],
    [-1.26, 1.03, 0.5],
    [0.64, 1.41, 0.5],
    [0.02, -0.46, 0.5],
    [0.29, -1.73, 0.5]
]

for i in range(len(mid_waypts)):
    mid_waypts[i] = pos_cal(mid_waypts[i], 0)

opti_start = [-1.49, -1.48, 0.5]
opti_target = [1.41, 0.73, 0.3]

def run():
    """
    The main function for one simulation test.
    """
    cf_settings = os.path.abspath(os.path.dirname(__file__)) + "/crazyflies.yaml"
    swarm = Crazyswarm(crazyflies_yaml=cf_settings)
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    # Load configuration.
    # CONFIG_FACTORY = ConfigFactory()
    # config = CONFIG_FACTORY.merge()

    CTRL_FREQ = 25
    EPISODE_LEN_SEC = 40
    
    # Height field load
    # imgpath = os.path.abspath(os.path.dirname(__file__)) + 'utils/map_generate/pictures/test.png'
    # config.quadrotor_config.obstacles['grid'] = constructHeightFieldDictFromImg(imgpath, 3., cell_size=0.1)

    start_pos = pos_cal(opti_start, 0)
    ref_pos = pos_cal(opti_target, 0)
    obs = [start_pos[0], 0, start_pos[1], 0, start_pos[2], 0, 0, 0, 0, 0, 0, 0]
    info = {"ctrl_timestep": 0.01, "ctrl_freq": 100, "episode_len_sec": 20, 
            "nominal_obstacles_info": obs_dict, "site_size": [3.8, 3.8, 0.9],
            "x_reference": [ref_pos[0], 0, ref_pos[1], 0, ref_pos[2], 0, 0, 0, 0, 0, 0, 0],
            "mid_waypoints": mid_waypts}
    
    
    resample = False
    # Create a logger and counters
    logger = Logger(logging_freq_hz=CTRL_FREQ)

    # obs = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r}
    commander = Planner(obs, info, verbose=False, logger=None)

    cf.goTo(opti_start, 0., duration=5)
    timeHelper.sleep(5)

    curr_pos = cf.position()
    curr_pos = pos_cal(curr_pos, 0)
    curr_x = curr_pos[0]
    curr_y = curr_pos[1]
    curr_z = curr_pos[2]

    first_ep_iteration = True
    ep_start = timeHelper.time()
    last_time = 0
    for i in range(CTRL_FREQ*EPISODE_LEN_SEC):

        # Elapsed sim time.
        # curr_time = (i-episode_start_iter)*CTRL_DT
        curr_time = timeHelper.time() - ep_start
        dt = curr_time - last_time
        last_time = curr_time
        # Compute control input.
        if first_ep_iteration:
            done = False
            info = {}
            first_ep_iteration = False

        last_x = curr_x
        last_y = curr_y
        last_z = curr_z

        curr_pos = cf.position()
        curr_pos = pos_cal(curr_pos, 0)
        #  quat rotation as (x,y,z,w)
        curr_x = curr_pos[0]
        curr_y = curr_pos[1]
        curr_z = curr_pos[2]
        if i == 0:
            vel = [0, 0, 0]
        else:
            vel = diff([curr_x, curr_y, curr_z], [last_x, last_y, last_z], dt)

        # rotate = quat2eulers(quat[3], quat[0], quat[1], quat[2])
        rotate = quat2eulers(1., 0., 0., 0.)
        obs = [curr_x, vel[0], curr_y, vel[1], curr_z, vel[2], 
                rotate[0], rotate[1], rotate[2], 0, 0, 0]
        target_pos, target_vel, target_acc, target_omega, target_yaw = commander.cmdSimulation(curr_time, obs, resample=resample, info=info)

        target_pos = pos_cal(target_pos, 1)
        print("Traget Pos:", target_pos)
        cf.cmdFullState(target_pos, target_vel, target_acc, target_yaw, target_omega)
        
        # If an episode is complete, reset the environment.
        target = [target_pos[0], target_pos[1], target_pos[2], target_vel[0], target_vel[1], target_vel[2],
                  target_acc[0], target_acc[1], target_acc[2], target_yaw, target_omega[0], target_omega[1], target_omega[2]]
        logger.log(0, curr_time, obs, target)
        timeHelper.sleep(0.04)
        if (abs(ref_pos[0] - curr_x) <= 0.03) and (abs(ref_pos[1] - curr_y) <= 0.03) and (abs(ref_pos[2] - curr_z) <= 0.05):
            logger.save(new_feature=resample)
            break
    # Close the environment and print timing statistics.

    cf.notifySetpointsStop()
    cf.land(0.06, duration=5)



if __name__ == "__main__":
    run()