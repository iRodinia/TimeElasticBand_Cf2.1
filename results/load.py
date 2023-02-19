import os
import numpy as np

abs_path = os.path.abspath(os.path.dirname(__file__))
filename = abs_path + "/with_resampler-02.19.2023_15.03.40" + ".npy"

data_dict = np.load(filename, allow_pickle=True)

timestamps = data_dict['timestamps'][0]
states = data_dict['states'][0]
targets = data_dict['targets'][0]

a=0