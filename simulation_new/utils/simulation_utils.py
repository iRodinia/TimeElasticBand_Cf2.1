"""IROS 2022 competition utility classes and functions.

"""
import os
import time
import pybullet as p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from enum import Enum
from functools import wraps


class Command(Enum):
    """Command types that can be used with pycffirmware.

    """
    FINISHED = -1 # Args: None (exits the control loop)
    NONE = 0 # Args: None (do nothing)

    FULLSTATE = 1 # Args: [pos, vel, acc, yaw, rpy_rate] 
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.cmdFullState
    TAKEOFF = 2 # Args: [height, duration]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.takeoff
    LAND = 3 # Args: [height, duration]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.land
    STOP = 4 # Args: None
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.stop
    GOTO = 5 # Args: [pos, yaw, duration, relative (bool)]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.goTo

    NOTIFYSETPOINTSTOP = 6 # Args: None
        # Must be called to transfer drone state from low level control (cmdFullState) to high level control (takeoff, land, goto)
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.notifySetpointsStop


def timing_step(function):
    """Interstep learning timing decorator.

    """
    @wraps(function)
    def wrap(*args, **keyword_args):
        start = time.time()
        result = function(*args, **keyword_args)
        elapsed = time.time()-start
        args[0].interstep_learning_time += elapsed
        args[0].interstep_learning_occurrences += 1
        if elapsed >= args[0].CTRL_TIMESTEP:
            print('\n[WARNING] Function "{}" took: {} sec (too slow)'.format(function.__name__, elapsed))
        if args[0].VERBOSE and args[0].interstep_counter%int(args[0].CTRL_FREQ/2) == 0:
            print('\n{}-th call to function "{}" took: {} sec'.format(args[0].interstep_counter, function.__name__, elapsed))
        return result
    return wrap


def timing_ep(function):
    """Interepisode learning timing decorator.

    """
    @wraps(function)
    def wrap(*args, **keyword_args):
        start = time.time()
        result = function(*args, **keyword_args)
        elapsed = time.time()-start
        args[0].interepisode_learning_time = elapsed
        if args[0].VERBOSE:
            print('\n{}-th call to function "{}" took: {} sec'.format(args[0].interepisode_counter, function.__name__, elapsed))
        return result
    return wrap

def plot_trajectory(t_scaled,
                    waypoints,
                    ref_x,
                    ref_y,
                    ref_z
                    ):
    """Plot the trajectory with matplotlib.

    """
    # Plot each dimension.
    _, axs = plt.subplots(3, 1)
    axs[0].plot(t_scaled, ref_x)
    axs[0].set_ylabel('x (m)')
    axs[1].plot(t_scaled, ref_y)
    axs[1].set_ylabel('y (m)')
    axs[2].plot(t_scaled, ref_z)
    axs[2].set_ylabel('z (m)')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Plot in 3D.
    ax = plt.axes(projection='3d')
    ax.plot3D(ref_x, ref_y, ref_z)
    ax.scatter3D(waypoints[:,0], waypoints[:,1], waypoints[:,2])
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def draw_trajectory(waypoints,
                    ref_x,
                    ref_y,
                    ref_z):
    """Draw a trajectory in PyBullet's GUI.

    """
    fig = plt.figure("Traj")
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection='3d')
    ax.plot(ref_x, ref_y, ref_z)
    plt.savefig('savefig_example.png')
    plt.show()