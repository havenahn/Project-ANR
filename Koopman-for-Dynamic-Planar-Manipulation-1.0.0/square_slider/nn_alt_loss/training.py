
#!/usr/bin/env python
import helpers.learn_modules as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time as time


if __name__== "__main__":
    N_states = 6
    num_obs = 100

    # data = np.load('../data/random/Fx_3_13700274_combined.npy')
    # data = data[::5, :,2:]
    data = np.load('../data/random/Fx_3_4650093_combined.npy')
    data = data[:,:,2:]
    print("Data shape:", data.shape)
    lrn = lm.L3(N_x = N_states, N_z = 32, N_e = num_obs, epochs = 1000, batch_size = 8192)
    t0 = time.time()
    lrn.learn(data)
    print("Time elapsed:")
    print(time.time()-t0)