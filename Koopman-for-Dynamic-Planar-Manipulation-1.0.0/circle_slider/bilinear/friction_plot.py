import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

from planar_pusher import Pusher

def multi_plot_friction(pusher, state_data0, state_data1, state_data2, reference_traj=None, save = False):
    plt.rcParams['text.usetex'] = True
    # if reference_traj is None:
    #     raise ValueError('Reference trajectory not provided')
    state_data0_xy = np.array(pusher.Convert_to_xy(state_data0))
    state_data1_xy = np.array(pusher.Convert_to_xy(state_data1))
    state_data2_xy = np.array(pusher.Convert_to_xy(state_data2))
    comparison_array = reference_traj

    # Create a plot of the xy position of the slider, with the three different trajectories
    fig = plt.figure(figsize=(12, 6))
    plt.plot(state_data0_xy[:,0], state_data0_xy[:,1], label='Small Friction', color='blue', linestyle='--')
    plt.plot(state_data1_xy[:,0], state_data1_xy[:,1], label='True Friction', color='orange', linestyle=':')
    plt.plot(state_data2_xy[:,0], state_data2_xy[:,1], label='High Friction', color='green', linestyle='-.')
    plt.legend()
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Slider Trajectories with Different Frictions')
    if save:
        plt.savefig('results/sliderfriction_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()

state_data0 = np.load('results/SmallishFric_3_state_data.npy')
# state_data1 = np.load('results/RightFriction_3_state_data.npy')
# state_data2 = np.load('results/BigFriction_3_state_data.npy')
SAVE = True
num_states = 4
dt = 0.1

max_timesteps = 200

conduit_traj = np.zeros((max_timesteps,num_states+2))
conduit_traj[:,0] = np.arange(0,max_timesteps)*0.1*dt
conduit_traj[:,0] = np.arange(0,max_timesteps)*0.015*dt
conduit_traj[:,0] = np.arange(0,max_timesteps)*0.005*dt
# Set initial values
mu_p = 0.3
mu_t = 0.35
mass_block = 0.827
block_length = 0.09
pusher_k = 1000
max_energy = None
init_state = np.array(
    [0, 0, block_length/2, np.deg2rad(180), 0, 0])
pusher = Pusher(block_length, mass_block, mu_t, mu_p, pusher_k, init_state)
multi_plot_friction(pusher, state_data0, state_data1, state_data2, conduit_traj, save=SAVE)