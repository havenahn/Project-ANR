
#!/usr/bin/env python
import helpers.learn_modules as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time as time


if __name__== "__main__":
    N_states = 4
    num_obs = 100

    data = np.load('../data/random/Fx_5_5350107_combined.npy')
    print("Data shape:", data.shape)
    controlled_data = data[:, :3,2:]
    print("Controlled data shape:", controlled_data.shape)
    lrn = lm.L3(N_x = N_states, N_z = 32, N_e = num_obs, epochs = 300, batch_size = 8192)
    t0 = time.time()
    lrn.learn(controlled_data)
    print("Time elapsed:")
    print(time.time()-t0)