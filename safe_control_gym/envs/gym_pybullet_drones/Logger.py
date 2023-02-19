import os
from datetime import datetime
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Logger(object):
    """A class for logging and visualization.

    Stores, saves to file, and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(self,
                 logging_freq_hz: int,
                 output_folder: str="results",
                 num_drones: int=1,
                 duration_sec: int=0,
                 colab: bool=False,
                 ):
        """Logger class __init__ method.

        Note: the order in which information is stored by Logger.log() is not the same
        as the one in, e.g., the obs["id"]["state"], check the implementation below.

        Parameters
        ----------
        logging_freq_hz : int
            Logging frequency in Hz.
        num_drones : int, optional
            Number of drones.
        duration_sec : int, optional
            Used to preallocate the log arrays (improves performance).

        """
        self.NUM_STATE = 12
        self.NUM_TARGET = 13

        self.COLAB = colab
        self.OUTPUT_FOLDER = output_folder
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.mkdir(self.OUTPUT_FOLDER)
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.NUM_DRONES = num_drones
        self.PREALLOCATED_ARRAYS = False if duration_sec == 0 else True
        self.counters = np.zeros(num_drones)
        self.timestamps = np.zeros((num_drones, duration_sec*self.LOGGING_FREQ_HZ))

        self.states = np.zeros((num_drones, self.NUM_STATE, duration_sec*self.LOGGING_FREQ_HZ))
        # 16 states: 
        # (0)pos_x, (1)pos_y, (2)pos_z,
        # (3)vel_x, (4)vel_y, (5)vel_z,
        # (6)roll, (7)pitch, (8)yaw,
        # (9)ang_vel_x, (10)ang_vel_y, (11)ang_vel_z,
        # (12)rpm0, (13)rpm1, (14)rpm2, (15)rpm3
        self.targets = np.zeros((num_drones, self.NUM_TARGET, duration_sec*self.LOGGING_FREQ_HZ))
        # 13 targets: 
        # (0)pos_x, (1)pos_y, (2)pos_z,
        # (3)vel_x, (4)vel_y, (5)vel_z,
        # (6)acc_x, (7)acc_y, (8)acc_z,
        # (9)ang_vel_x, (10)ang_vel_y, (11)ang_vel_z,
        # (12)yaw
        
        # self.controls = np.zeros((num_drones, 4, duration_sec*self.LOGGING_FREQ_HZ))
        # 4 control inputs (Force, in N):
        # (0)f1, (1)f2, (2)f3, (3)f4 

        self.data_dict = {}
        # Other data to be logged
        # In the form: "data_name": data_array

    ################################################################################

    def log(self,
            drone: int,
            timestamp,
            state,
            target
            ):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        drone : int
            Id of the drone associated to the log entry.
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (20,)-shaped array of floats containing the drone's state.
        target : ndarray
            (12,)-shaped array of floats containing the drone's reference target.
        control : ndarray
            (4,)-shaped array of floats containing the drone's control input.
        """
        if drone < 0 or drone >= self.NUM_DRONES or timestamp < 0 or len(state) != self.NUM_STATE or len(target) != self.NUM_TARGET:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        #### Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, self.NUM_STATE, 1))), axis=2)
            self.targets = np.concatenate((self.targets, np.zeros((self.NUM_DRONES, self.NUM_TARGET, 1))), axis=2)
            # self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 4, 1))), axis=2)
        #### Advance a counter is the matrices have overgrown it ###
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter:
            current_counter = self.timestamps.shape[1]-1
        #### Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        #### Re-order the kinematic obs (of most Aviaries) #########
        self.states[drone, :, current_counter] = state
        self.targets[drone, :, current_counter] = target
        # self.controls[drone, :, current_counter] = control
        self.counters[drone] = current_counter + 1

    ################################################################################

    def log_extend(self, data_name: str, data):
        if data_name in self.data_dict:
            self.data_dict[data_name].append(data)
        else:
            self.data_dict[data_name] = [data]
    
    ################################################################################

    def save(self, new_feature=True, filename: str=None):
        """Save the logs to file.
        """
        if new_feature:
            file_name_mark = "with_resampler-"
        else:
            file_name_mark = "no_resampler-"

        if filename is not None:
             with open(os.path.join(self.OUTPUT_FOLDER, filename+".npy"), 'wb') as out_file:
                np.savez(
                    out_file,
                    timestamps=self.timestamps,
                    states=self.states,
                    targets=self.targets,
                    # controls=self.controls,
                    other_data=self.data_dict
                )
        else:
            with open(os.path.join(self.OUTPUT_FOLDER, file_name_mark+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".npy"), 'wb') as out_file:
                np.savez(
                    out_file,
                    timestamps=self.timestamps,
                    states=self.states,
                    targets=self.targets,
                    # controls=self.controls,
                    other_data=self.data_dict
                )

    ################################################################################
    
    def plot(self,
             comment: str="",
             pwm=False,
             autoclose=True):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        pwm : bool, optional
            If True, converts logged RPM into PWM values (for Crazyflies).

        """
        #### Loop over colors and line styles ######################
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, axs = plt.subplots(10, 2)
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)
        t = t[:len(self.states[0, 0, :])]

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 0, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 1, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')

        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 2, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')

        #### RPY ###################################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 6, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('phi (rad)')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 7, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('theta (rad)')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 8, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('psi (rad)')

        #### BF Rates ##############################################
        # row = 6
        # for j in range(self.NUM_DRONES):
        #     axs[row, col].plot(t, self.states[j, 9, :], label="drone_"+str(j))
        # axs[row, col].set_xlabel('time')
        # axs[row, col].set_ylabel('p (rad/s)')
        # row = 7
        # for j in range(self.NUM_DRONES):
        #     axs[row, col].plot(t, self.states[j, 10, :], label="drone_"+str(j))
        # axs[row, col].set_xlabel('time')
        # axs[row, col].set_ylabel('q (rad/s)')
        # row = 8
        # for j in range(self.NUM_DRONES):
        #     axs[row, col].plot(t, self.states[j, 11, :], label="drone_"+str(j))
        # axs[row, col].set_xlabel('time')
        # axs[row, col].set_ylabel('r (rad/s)')
        row = 6
        axs[row, col].plot(t, t, label="time")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('time')
        row = 7
        axs[row, col].plot(t, t, label="time")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('time')
        row = 8
        axs[row, col].plot(t, t, label="time")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('time')

        #### Time ##################################################
        row = 9
        axs[row, col].plot(t, t, label="time")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('time')

        #### Column ################################################
        col = 1

        #### Velocity ##############################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 3, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 4, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 5, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        #### RPY Rates #############################################
        # row = 3
        # for j in range(self.NUM_DRONES):
        #     rdot = np.hstack([0, (self.states[j, 6, 1:] - self.states[j, 6, 0:-1]) * self.LOGGING_FREQ_HZ ])
        #     axs[row, col].plot(t, rdot, label="drone_"+str(j))
        # axs[row, col].set_xlabel('time')
        # axs[row, col].set_ylabel('rdot (rad/s)')
        # row = 4
        # for j in range(self.NUM_DRONES):
        #     pdot = np.hstack([0, (self.states[j, 7, 1:] - self.states[j, 7, 0:-1]) * self.LOGGING_FREQ_HZ ])
        #     axs[row, col].plot(t, pdot, label="drone_"+str(j))
        # axs[row, col].set_xlabel('time')
        # axs[row, col].set_ylabel('pdot (rad/s)')
        # row = 5
        # for j in range(self.NUM_DRONES):
        #     ydot = np.hstack([0, (self.states[j, 8, 1:] - self.states[j, 8, 0:-1]) * self.LOGGING_FREQ_HZ ])
        #     axs[row, col].plot(t, ydot, label="drone_"+str(j))
        # axs[row, col].set_xlabel('time')
        # axs[row, col].set_ylabel('ydot (rad/s)')

        #### BF Rates ##############################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 9, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p (rad/s)')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 10, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('q (rad/s)')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 11, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r (rad/s)')

        ### This IF converts RPM into PWM for all drones ###########
        #### except drone_0 (only used in examples/compare.py) #####
        for j in range(self.NUM_DRONES):
            for i in range(12,16):
                if pwm and j > 0:
                    self.states[j, i, :] = (self.states[j, i, :] - 4070.3) / 0.2685

        #### RPMs ##################################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 12, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM0')
        else:
            axs[row, col].set_ylabel('RPM0')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 13, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM1')
        else:
            axs[row, col].set_ylabel('RPM1')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 14, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM2')
        else:
            axs[row, col].set_ylabel('RPM2')
        row = 9
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 15, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM3')
        else:
            axs[row, col].set_ylabel('RPM3')

        #### Drawing options #######################################
        for i in range (10):
            for j in range (2):
                axs[i, j].grid(True)
                axs[i, j].legend(loc='upper right',
                         frameon=True
                         )
        fig.subplots_adjust(left=0.06,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.15,
                            hspace=0.0
                            )
        plt.savefig(os.path.join('results', comment+'-output_figure.png'))
        if autoclose:
            plt.show(block=False)
            plt.pause(2)
            plt.close()
        else:
            plt.show()
