import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

from planar_pusher import Pusher



class Plotter:
    def __init__(self, pusher, state_data, reference_traj = None, save = False):
        self.pusher = pusher
        self.state_data = state_data
        self.state_data_xy = pusher.Convert_to_xy(state_data)
        self.reference_traj = reference_traj
        self.save = save

    def plot_traj_stretch_final(self, stretch_factor = 2.0, plot_indices = range(0,100, 10)):
        plt.rcParams['text.usetex'] = True
        if self.reference_traj is None:
            raise ValueError('Reference trajectory not provided')

        state_array = self.state_data_xy
        comparison_array = self.reference_traj
        # if self.save:
        #     plt.style.use(['science'])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True) 


        def add_frame(plotter, frame, state_array, comparison_array):
            """Function to animate the pusher."""
            current_state = state_array[frame,:]
            fade = 0.8 - (frame / len(state_array))*(0.8)
            if fade <= 0.1:
                fade = 0.1
            
            block_patch = plt.Circle((current_state[0], current_state[1]), self.pusher.block_diameter/2, fc='b', alpha=fade, ec='k', linestyle='--', lw=1.)
            pusher_patch = plt.Circle([current_state[2], current_state[3]], self.pusher.block_diameter/20, fc='r',zorder=2, alpha=0.8, ec='k', linestyle='--', lw=0.1)

            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)

            # comparison_state = comparison_array[frame]
            # comparison_block_patch.set_center([comparison_state[0], comparison_state[1]])

        def add_frame_init(plotter, frame, state_array, comparison_array):
            """Function to animate the pusher."""
            current_state = state_array[frame,:]
            fade = 0.8 - (frame / len(state_array))*(0.8)
            if fade <= 0.1:
                fade = 0.1
            
            block_patch = plt.Circle((current_state[0], current_state[1]), self.pusher.block_diameter/2, fc='b', alpha=fade, ec='k', linestyle='--', lw=1., label='Slider')
            pusher_patch = plt.Circle([current_state[2], current_state[3]], self.pusher.block_diameter/20, fc='r',zorder=2, alpha=fade, ec='k', linestyle='--', lw=0.1, label='Pusher')

            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)
            return block_patch, pusher_patch

            # comparison_state = comparison_array[frame]
            # comparison_block_patch.set_center([comparison_state[0], comparison_state[1]])
        
        lims = [-0.07,0.2,-0.1,0.1]
        lims = [-0.1,0.22,-0.1,0.1]
        lims = [-0.1,0.5,-0.15,0.2]
        # Add a lineplot to the figure showing how the pusher evolves
        state_array_np = np.array(state_array)
        # Compute stretched x-values
        dx = np.diff(state_array_np[:,0])  # Compute differences between adjacent x-values
        stretched_dx = dx * stretch_factor  # Scale the differences
        stretched_x_values = np.concatenate(([state_array_np[0,0]], state_array_np[0,0] + np.cumsum(stretched_dx)))

        # Compute stretched px-values
        # Compute shift for each timestep
        shift = stretched_x_values - state_array_np[:,0]  # How much each point moved

        # Apply the same shift to x2-values
        stretched_px_values = state_array_np[:,2] + shift

        stretched_state_array = np.copy(state_array_np)
        stretched_state_array[:,0] = stretched_x_values
        stretched_state_array[:,2] = stretched_px_values
        for i in plot_indices:
            if i == 0:
                # Add initial with labels
                block_patch, pusher_patch = add_frame_init(self, i, stretched_state_array, comparison_array)
            add_frame(self, i, stretched_state_array, comparison_array)
        add_frame(self, stretched_state_array.shape[0]-1, stretched_state_array, comparison_array)

        # Add a lineplot to the figure showing how the pusher evolves
        c1 = plot_indices[0]
        c2 = plot_indices[1]
        c_arr = self.contact_switches

        # First segment no-contact:
        l_arr = []
        # l1 = ax.plot(stretched_state_array[c1:c2,2], stretched_state_array[c1:c2,3], color='r', linestyle='--',label="Pusher Trajectory: No Contact")
        # l_arr.append(l1)
        for i in range(0,len(c_arr)-1):
            if c_arr[i][1] == 1:
                l = ax.plot(stretched_state_array[c_arr[i][0]:c_arr[i+1][0],2], stretched_state_array[c_arr[i][0]:c_arr[i+1][0],3], color='r', linestyle='-',label="Pusher Trajectory: Contact")
                l_arr.append(l)
            else:
                l = ax.plot(stretched_state_array[c_arr[i][0]:c_arr[i+1][0],2], stretched_state_array[c_arr[i][0]:c_arr[i+1][0],3], color='r', linestyle='--',label="Pusher Trajectory: No Contact")
                l_arr.append(l)
        if c_arr[-1][1] == 1:
            l = ax.plot(stretched_state_array[c_arr[-1][0]:,2], stretched_state_array[c_arr[-1][0]:,3], color='r', linestyle='-',label="Pusher Trajectory: Contact")
            l_arr.append(l)
        else:
            l = ax.plot(stretched_state_array[c_arr[-1][0]:,2], stretched_state_array[c_arr[-1][0]:,3], color='r', linestyle='--',label="Pusher Trajectory: No Contact")
            l_arr.append(l)

        # Second segment contact:
        #l2 = ax.plot(stretched_state_array[c2:,2], stretched_state_array[c2:,3], color='r', linestyle='-',label="Pusher Trajectory: Contact")
        ax.plot()
        plt.title(r"$\textrm{Pusher Repositioning for a Rightward Trajectory}$")
        plt.xlabel(r"$\textrm{X Position (m)}$")
        plt.ylabel(r"$\textrm{Y Position (m)}$")
        #plt.legend()
        plt.grid(False) 
        # Add arrows at regular intervals
        arrow_spacing = 20  # Place an arrow every 2 points
        for i in range(0, stretched_state_array.shape[0] - 1, arrow_spacing):
            plt.annotate("",
                        xy=(stretched_state_array[i+1,2], stretched_state_array[i+1,3]),  # Arrowhead
                        xytext=(stretched_state_array[i,2], stretched_state_array[i,3]),  # Arrow tail
                        arrowprops=dict(arrowstyle="->", lw=1, color="red"))
        if lims is None:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
            ax.set_aspect('equal', adjustable='box')
        if self.save:
            plt.savefig('results/smallishfric_stretched_traj.png', dpi=600)
        else:
            plt.show()

    def plot_traj_stretch(self, stretch_factor = 2.0, plot_indices = range(0,100, 10)):
        plt.rcParams['text.usetex'] = True
        if self.reference_traj is None:
            raise ValueError('Reference trajectory not provided')

        state_array = self.state_data_xy
        comparison_array = self.reference_traj
        # if self.save:
        #     plt.style.use(['science'])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True) 


        def add_frame(plotter, frame, state_array, comparison_array):
            """Function to animate the pusher."""
            current_state = state_array[frame,:]
            fade = 0.8 - (frame / len(state_array))*(0.8)
            if fade <= 0.1:
                fade = 0.1
            
            block_patch = plt.Circle((current_state[0], current_state[1]), self.pusher.block_diameter/2, fc='b', alpha=fade, ec='k', linestyle='--', lw=1.)
            pusher_patch = plt.Circle([current_state[2], current_state[3]], self.pusher.block_diameter/20, fc='r',zorder=2, alpha=fade, ec='k', linestyle='--', lw=0.1)

            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)

            # comparison_state = comparison_array[frame]
            # comparison_block_patch.set_center([comparison_state[0], comparison_state[1]])

        def add_frame_init(plotter, frame, state_array, comparison_array):
            """Function to animate the pusher."""
            current_state = state_array[frame,:]
            fade = 0.8 - (frame / len(state_array))*(0.8)
            if fade <= 0.1:
                fade = 0.1
            
            block_patch = plt.Circle((current_state[0], current_state[1]), self.pusher.block_diameter/2, fc='b', alpha=fade, ec='k', linestyle='--', lw=1., label='Slider')
            pusher_patch = plt.Circle([current_state[2], current_state[3]], self.pusher.block_diameter/20, fc='r',zorder=2, alpha=fade, ec='k', linestyle='--', lw=0.1, label='Pusher')

            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)
            return block_patch, pusher_patch

            # comparison_state = comparison_array[frame]
            # comparison_block_patch.set_center([comparison_state[0], comparison_state[1]])
        
        lims = [-0.07,0.2,-0.1,0.1]
        lims = [-0.1,0.22,-0.1,0.1]
        # Add a lineplot to the figure showing how the pusher evolves
        state_array_np = np.array(state_array)
        # Compute stretched x-values
        dx = np.diff(state_array_np[:,0])  # Compute differences between adjacent x-values
        stretched_dx = dx * stretch_factor  # Scale the differences
        stretched_x_values = np.concatenate(([state_array_np[0,0]], state_array_np[0,0] + np.cumsum(stretched_dx)))

        # Compute stretched px-values
        # Compute shift for each timestep
        shift = stretched_x_values - state_array_np[:,0]  # How much each point moved

        # Apply the same shift to x2-values
        stretched_px_values = state_array_np[:,2] + shift

        stretched_state_array = np.copy(state_array_np)
        stretched_state_array[:,0] = stretched_x_values
        stretched_state_array[:,2] = stretched_px_values
        for i in plot_indices:
            if i == 0:
                # Add initial with labels
                block_patch, pusher_patch = add_frame_init(self, i, stretched_state_array, comparison_array)
            add_frame(self, i, stretched_state_array, comparison_array)
        add_frame(self, stretched_state_array.shape[0]-1, stretched_state_array, comparison_array)

        
        ax.plot(stretched_state_array[:,2], stretched_state_array[:,3], color='r', linestyle='--',label="Pusher Trajectory")
        plt.title(r"$\textrm{Following a Horizontal Trajectory}$")
        plt.xlabel(r"$\textrm{X Position (m)}$")
        plt.ylabel(r"$\textrm{Y Position (m)}$")
        plt.legend()
        plt.grid(False) 
        # Add arrows at regular intervals
        arrow_spacing = 20  # Place an arrow every 2 points
        for i in range(0, stretched_state_array.shape[0] - 1, arrow_spacing):
            plt.annotate("",
                        xy=(stretched_state_array[i+1,2], stretched_state_array[i+1,3]),  # Arrowhead
                        xytext=(stretched_state_array[i,2], stretched_state_array[i,3]),  # Arrow tail
                        arrowprops=dict(arrowstyle="->", lw=1, color="red"))
        if lims is None:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
            ax.set_aspect('equal', adjustable='box')
        if self.save:
            plt.savefig('results/stretched_traj.png', dpi=600)
        else:
            plt.show()

    def plot_traj_wref(self, interval = 10):
        if self.reference_traj is None:
            raise ValueError('Reference trajectory not provided')

        state_array = self.state_data_xy
        comparison_array = self.reference_traj
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True) 


        def add_frame(plotter, frame, state_array, comparison_array):
            """Function to animate the pusher."""
            current_state = state_array[frame]
            fade = 0.8 - (frame / len(state_array))*(0.8)
            
            block_patch = plt.Circle((current_state[0], current_state[1]), self.pusher.block_diameter/2, fc='b', alpha=fade, ec='k', linestyle='--')
            pusher_patch = plt.Circle([current_state[2], current_state[3]], self.pusher.block_diameter/20, fc='r',zorder=2, alpha=fade, ec='k', linestyle='--')

            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)

            # comparison_state = comparison_array[frame]
            # comparison_block_patch.set_center([comparison_state[0], comparison_state[1]])
        
        lims = [-0.1,0.2,-0.1,0.1]
        for i in range(0, len(state_array), interval):
            add_frame(self, i, state_array, comparison_array)
        # Add a lineplot to the figure showing how the pusher evolves
        state_array_np = np.array(state_array)
        ax.plot(state_array_np[:,2], state_array_np[:,3], color='r', linestyle='--')
        if lims is None:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
            ax.set_aspect('equal', adjustable='box')
        plt.show()

    def find_contact_switches(self):
        """Finds the contact switches in the state data."""
        contact_switches = []
        contact_dist = self.state_data[:,2].copy() - self.pusher.block_diameter/2
        for i in range(1, len(self.state_data)):
            if contact_dist[i] > 0 and contact_dist[i-1] <= 0:
                contact_switches.append([i, -1])
            elif contact_dist[i] <= 0 and contact_dist[i-1] > 0:
                contact_switches.append([i, 1])
        return contact_switches
        

# Load the data
#state_data = np.load('results/PusherRepositioning_3_state_data.npy')
#state_data = np.load('results/BigFriction_3_state_data.npy')
state_data = np.load('results/SmallishFric_3_state_data.npy')
data = 
#state_data = np.load('results/RightFriction_3_state_data.npy')

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
block_length = 0.16
pusher_k = 1000
max_energy = None
init_state = np.array(
    [0, 0, block_length/2, np.deg2rad(180), 0, 0])
pusher = Pusher(block_length, mass_block, mu_t, mu_p, pusher_k, init_state)
plotter = Plotter(pusher, state_data, conduit_traj)
contact_switches = plotter.find_contact_switches() # Could use these to modify plot, but there are some brief switches
print("Contact switches: ", contact_switches)
plotter.contact_switches = contact_switches
plotter.save = SAVE
#plotter.plot_traj_wref(interval=50)
# Indices to plot: 0, 33, 89, 111, 150, end (auto-include)
#plotter.plot_traj_stretch(stretch_factor=2.0, plot_indices=range(0,len(state_data), 50))
plotter.plot_traj_stretch_final(stretch_factor=2.0, plot_indices=[0,50,85,95,141,261,324,399])


