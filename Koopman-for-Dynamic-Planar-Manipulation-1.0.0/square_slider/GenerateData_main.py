from planar_pusher import Pusher
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.integrate as spi
#import spatialmath as sm
import time
import sys
import argparse
import gc
from GenerateData_subset import generate_data
import multiprocessing

# Generate a series of data points for the planar pusher model based on trajectory data.

# Define the number of data points to generate

if __name__ == '__main__':
    """
    t: time 
    state: state vector [x, y, theta, px, py, x_dot, y_dot, theta_dot] 
    u: control input [px_dot, py_dot]
    """

    #manager = multiprocessing.Manager()
    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Generate data for the planar pusher model.')
    parser.add_argument('--duration', type=float, default=1, help='Duration of data generation in hours.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
    parser.add_argument('--save', type=bool, default=True, help='Save the generated data.')
    args = parser.parse_args()
    duration = args.duration
    SAVE = args.save
    seed = args.seed

    traj_len = 3
    #SAVE = True
    save_path = 'data'
    num_states = 12

    mu_p = 0.3
    mu_t = 0.35
    mass_block = 0.029
    block_length = 0.064
    max_energy = None

    # For straight line test. The domains are relative to the trajectory.
    x_domain = np.array([-block_length/20, block_length/20])
    y_domain = np.array([-block_length/20, block_length/20])
    theta_domain = np.array([-np.deg2rad(20), np.deg2rad(20)])
    pbody1_domain = np.array([-0.011,0.011])
    #pbody2_domain = np.array([-0.9*block_length/2, 0.9*block_length/2])
    pbody2_domain = np.array([-block_length/10, block_length/10])
    x_dot_domain = np.array([-0.05, 0.05])
    y_dot_domain = np.array([-0.05, 0.05])
    theta_dot_domain = np.array([-0.05, 0.05])
    # BC_body1_domain = pbody1_domain
    # BC_body2_domain = pbody2_domain
    # BC_body1_dot_domain = np.array([-0.005, 0.005])
    # BC_body2_dot_domain = np.array([-0.005, 0.005])

    domains = [x_domain, y_domain, theta_domain, pbody1_domain, pbody2_domain, x_dot_domain, y_dot_domain, theta_dot_domain]

    #dt = 0.01
    dt = 0.1
    conduit_traj = np.load('pusher_trajectory.npy')
    init_state = np.array(
        [0, 0, 0, -block_length, 0, 0, 0, 0]
    )  # [x, y, theta, px, py, x_dot, y_dot, theta_dot]
    pusher = Pusher(block_length, mass_block, mu_t, mu_p, init_state)
    start_time = time.time()
    loop_counter = 0
    while time.time() - start_time < duration*60*60:
        #generate_data(conduit_traj, max_energy, traj_len,pusher,save_path,num_states,dt,domains,SAVE,duration=duration,seed=seed, file_array_size=10000,file_count = loop_counter)
        p1 = multiprocessing.Process(target=generate_data, args=(conduit_traj, max_energy, traj_len,pusher,save_path,num_states,dt,domains,SAVE,False,duration,seed,50000,loop_counter))
        p1.start()
        print('Loop counter: ' + str(loop_counter))
        loop_counter += 1
        p1.join()

    # Plot the generated trajectories
    #data = np.array(data)
    #print(data.shape)