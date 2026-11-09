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

# Generate a series of data points for the planar pusher model based on trajectory data.

# Define the number of data points to generate

def generate_data(conduit_traj, max_energy,traj_len,pusher,save_path,num_states,dt, domains, SAVE=False,VERBOSE=False, duration=1.0,seed=0, file_array_size=10000,file_count=0):
    """Generates a series of data points for the planar pusher model. Random sampling is used to find data points within the domain."""
    low_limit = []
    upper_limit = []
    for domain in domains:
        low_limit.append(domain[0])
        upper_limit.append(domain[1])

    data = []
    rng = np.random.default_rng(seed)
    print('Seed:',seed)
    start_time = time.time()
    num_subsamples = 1000
    prev_time = start_time
    i = 0
    ctrl = np.array([0,0])

    # states [xg, yg, winch_rot, winch_rot_dot, x, y, z, psi, theta, phi, xdot, ydot, zdot, psidot, thetadot, phidot]
    #for i in range(num_samples):
    while i <= file_array_size:
        i += 1
        IN_CONTACT = rng.choice([True,False])
        state0_delta = rng.uniform(low=low_limit, high=upper_limit,size=None)
        conduit_traj_ind = rng.choice(np.arange(0,conduit_traj.shape[0]))
        state0 = conduit_traj[conduit_traj_ind,:]+state0_delta
        # Check if the random sample violates any bounds. If so, resample.
        min_px = -0.0391
        max_px = -0.0310
        assert min_px < max_px  
        REDO_COUNTER = 0
        MAX_REDO = 50
        X_dot = state0[5]
        Y_dot = state0[6]
        theta = state0[2]
        xy_to_body = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T
        x_dot, y_dot = np.array([X_dot, Y_dot]) @ xy_to_body
        if IN_CONTACT:
            px = rng.uniform(low=min_px, high=-pusher.block_length/2,size=None)
            state0[3] = px
        else:
            px = rng.uniform(low=-pusher.block_length/2, high=max_px,size=None)
            state0[3] = px
        if VERBOSE:
            print('State0: ' + str(state0))
        results_full_ivp = spi.solve_ivp(pusher.dynamics, method='LSODA',t_span=(0,traj_len*dt),t_eval=np.linspace(0,traj_len*dt,traj_len), y0=state0, args=(ctrl,))

        data.append(results)
        if i % num_subsamples == 0:
            print('Generating data for sample ' + str(i))
            print('Elapsed time: ' + str(time.time()-prev_time) + ' seconds')
            print('Average time per sample: ' + str((time.time()-prev_time)/num_subsamples) + ' seconds')
            prev_time = time.time()
                    
    num_samples = i
    print('Total Elapsed time: ' + str(time.time()-start_time) + ' seconds')
    print('Number of sample trajectories: ' + str(num_samples))
    print('Final value of i: ' + str(i))
    print('Seed:',seed)
    if SAVE:
        np.save(save_path+'/Fx_'+str(traj_len)+'_'+str(len(data))+'_'+str(seed)+'_'+'file'+str(file_count)+'.npy',data)
        print('Data saved to ' + save_path)
    return data

if __name__ == '__main__':
    """
    t: time 
    state: state vector [x, y, theta, px, py, x_dot, y_dot, theta_dot] 
    u: control input [px_dot, py_dot]
    """

    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Generate data for the planar pusher model.')
    parser.add_argument('--duration', type=float, default=1, help='Duration of data generation in hours.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
    parser.add_argument('--save', type=bool, default=True, help='Save the generated data.')
    args = parser.parse_args()
    duration = args.duration
    SAVE = args.save
    seed = args.seed

    traj_len = 10
    #SAVE = True
    save_path = 'data'
    num_states = 12

    mu_p = 0.3
    mu_t = 0.35
    mass_block = 0.827
    block_length = 0.09
    pusher_k = 1000
    max_energy = None

    # For straight line test. The domains are relative to the trajectory.
    x_domain = np.array([-block_length/20, block_length/20])
    y_domain = np.array([-block_length/20, block_length/20])
    theta_domain = np.array([-np.deg2rad(10), np.deg2rad(10)])
    pbody1_domain = np.array([-0.011,0.011])
    #pbody2_domain = np.array([-0.9*block_length/2, 0.9*block_length/2])
    pbody2_domain = np.array([-block_length/20, block_length/20])
    x_dot_domain = np.array([-0.05, 0.05])
    y_dot_domain = np.array([-0.05, 0.05])
    theta_dot_domain = np.array([-0.05, 0.05])
    # BC_body1_domain = pbody1_domain
    # BC_body2_domain = pbody2_domain
    # BC_body1_dot_domain = np.array([-0.005, 0.005])
    # BC_body2_dot_domain = np.array([-0.005, 0.005])

    domains = [x_domain, y_domain, theta_domain, pbody1_domain, pbody2_domain, x_dot_domain, y_dot_domain, theta_dot_domain]

    dt = 0.01
    conduit_traj = np.load('pusher_trajectory.npy')
    init_state = np.array(
        [0, 0, 0, -block_length, 0, 0, 0, 0]
    )  # [x, y, theta, px, py, x_dot, y_dot, theta_dot]
    pusher = Pusher(block_length, mass_block, mu_t, mu_p, pusher_k, init_state)
    data = generate_data(conduit_traj, max_energy, traj_len,pusher,save_path,num_states,dt,domains,SAVE,duration=duration,seed=seed)
    # Plot the generated trajectories
    data = np.array(data)
    print(data.shape)