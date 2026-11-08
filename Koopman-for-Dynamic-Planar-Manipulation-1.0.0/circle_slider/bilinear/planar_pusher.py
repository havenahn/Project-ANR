"""Module for the planar pusher dynamic model."""

import numpy as np
#import spatialmath as sm
import matplotlib.pyplot as plt
import scipy.integrate as spi
import matplotlib.animation as animation
import matplotlib as mpl
import math

SURFACE_F_ARR = []
BLOCK_F_ARR = []
FC_ARR = []


class Pusher:
    """Includes the functions to simulate the planar pusher model."""

    def __init__(self, block_diameter=1, block_mass=1, table_fric=10., pusher_mu=0.2, pusher_k=10, state=np.array([0,0,0,0,0,0,0,0,0,0,0,0])):
        self.block_diameter = block_diameter
        self.block_mass = block_mass
        self.table_fric = table_fric
        self.pusher_mu = pusher_mu
        self.g = 9.81
        self.pusher_k = pusher_k
        self.pusher_b = pusher_k/100
        self.mom_inertia = 0.5*self.block_mass*(self.block_diameter**2)/4
        self.state = state  # [x, y, theta, px, py, x_dot, y_dot, theta_dot]

    def dynamics_bodyu(self, t, state, u, DEBUG=False):
        """Returns the state derivative of the planar pusher model.
        Inputs:
            t: time 
            state: state vector [x, y, px, py, x_dot, y_dot, BC_body1, BC_body2,
            BC_body1_dot, BC_body2_dot] 
            u: control input [px_dot, py_dot]

        Outputs:
            state_dot: state derivative"""

        #print('t:',t)
        # Unpack state
        x, y, p_body1, p_body2, x_dot, y_dot = state
        p_body1_dot, p_body2_dot = u
        #p_body1_dot = -p_x_dot*np.cos(p_body2) - p_y_dot*np.sin(p_body2) + np.linalg.norm(np.array([x_dot,y_dot])) # Need to consider the motion of the block. Check if the input is in xy or local coords. I think it should be local to allow control coherent koopman
        #p_body2_dot = -(-p_x_dot*np.sin(p_body2) + p_y_dot*np.cos(p_body2))/p_body1 # Need to consider the motion of the block
        pusher_body = np.array([p_body1, p_body2])
        
        #BC_body = np.array([BC_body1, BC_body2])       
        block_xy = np.array([x, y])

        p_body_dot = np.array([p_body1_dot, p_body2_dot])

        # Determine the forces from the pusher
        # These are not considered in the case of a massless contact
        # pusher_f = -self.pusher_k * (C_xy - pusher_xy)
        # pusher_f += -self.pusher_b * (C_xy_dot - pusher_body_xy)
        # pusher_f_body = R.T @ pusher_f
        # fx = pusher_f_body[0] # force from spring
        # fy = pusher_f_body[1] # force from spring

        # Determine the contact forces
        # Assume that the pusher is always aligned rotationally with the local block coordinates
        pen_dist_c1 = -(pusher_body[0] - (self.block_diameter/2))
        #print('pusher body 0:',pusher_body[0])
        #print('pen_dist_c1:',pen_dist_c1)
        fc = 0 # fc is in the body frame
        if pen_dist_c1 > 0:
            #fc += -self.contact_k * pen_dist
            #fc += -self.contact_b * p_body_dot[0]
            fc += -self.pusher_k * pen_dist_c1


        # Determine the limits on the forces from the pusher on the block
        # Note that in this case, we assume that there is no friction from the pusher contact with the object!
        # max_push_f = abs(self.pusher_mu * fc) # max limit from friction cone
        # max_push_f is equal to zero due to the no-friction assumption!
        # if abs(fy) > max_push_f:
        #     fy_block = (
        #         np.sign(-fy) * max_push_f
        #     )  # set the force experienced by the block to lie on or within the friction cone
        # else:
        #     fy_block = -fy
        fnorm_block = -fc # This would be entirelt
        ftan_block = 0 # No friction from the pusher
        m_block = np.cross(
           [pusher_body[0],0], [fnorm_block, ftan_block]
        )  # This might be covered by the motion cone. Check this later.

        block_force_body = np.array([fnorm_block, ftan_block, m_block])
        R2 = np.array([[np.cos(-pusher_body[1]), np.sin(-pusher_body[1])],[-np.sin(-pusher_body[1]), np.cos(-pusher_body[1])]])
        block_force_xy = np.array([0,0,m_block])
        block_force_xy[:2] = R2 @ block_force_body[:2]
        #block_force_xy[:2] =  * fx_block
        #print('block_force_xy:',block_force_xy)
        #print('block_force_body:',block_force_body)
        #print('theta:',theta)
        surface_f = -self.table_fric*np.array([x_dot, y_dot, 0])

        # Determine the net force on the block
        net_f_block = np.zeros((3))
        #net_f_block[:2] = (surface_f[:2] + block_force[:2]) @ body_to_xy
        net_f_block[:2] = surface_f[:2] + block_force_xy[:2]
        #net_f_block[:2] = (surface_f[:2] + block_force[:2]) @ body_to_xy + np.cross([0,0,theta_dot], [BC_xy[0], BC_xy[1], 0])[:2]
        net_f_block[2] = surface_f[2] + block_force_xy[2]
        if net_f_block[2] != 0:
            raise ValueError('Net force in rotational direction is not zero!')
        # Determine the net force on the end effector. Due to massless pusher, this doesn't matter
        #net_f_ee = np.array([fc,fy]) @ body_to_xy + pusher_f

        # # Determine the forces on the pusher due to the quasi-static assumption, it is assumed that
        # # the friction from the ground fully balances out the forces being applied by the pusher's
        # # end effector. This means that the pusher's velocities remain constant? EXCEPT when sliding
        # # tangential to the object?
        # ee_y_f = (
        #     fy - fy_block
        # )  # Should only be non-zero when lying on the edge of the motion cone
        # # The below allows the pusher end effector to move away from the object. This may
        # # dramatically increase the complexity of the system, and we should probably start out with
        # # the assumption that this cannot happen.
        # if fx < 0:
        #     ee_x_f = fx
        # else:
        #     ee_x_f = x_dot + np.cross([0,0,theta_dot], [ex, ey, 0])[2]

        # Determine the state derivatives

        x_dt = x_dot
        y_dt = y_dot
        px_dt = p_body_dot[0]
        py_dt = p_body_dot[1]
        #x_dot_dt = block_xdt[0]
        #y_dot_dt = block_xdt[1]
        #theta_dot_dt = block_xdt[2]
        x_dot_dt = net_f_block[0] / self.block_mass
        y_dot_dt = net_f_block[1] / self.block_mass
        #BC_xy_dot_dt = net_f_ee - np.array([x_dot_dt, y_dot_dt])
        #BC_body_dot_dt = BC_xy_dot_dt @ xy_to_body
        #BC_body1_dt = BC_body1_dot
        #BC_body2_dt = BC_body2_dot
        #ex_dot_dt = ee_x_f
        #ey_dot_dt = ee_y_f
        #BC_body1_dot_dt = BC_body_dot_dt[0]
        #BC_body2_dot_dt = BC_body_dot_dt[1]
        # if x_dt != 0:
        #     print('moving block')
        state_dt = np.array(
            [
                x_dt,
                y_dt,
                px_dt,
                py_dt,
                x_dot_dt,
                y_dot_dt
                #BC_body1_dt,
                #BC_body2_dt,
                #BC_body1_dot_dt,
                #BC_body2_dot_dt,
            ]
        )

        SURFACE_F_ARR.append(surface_f)
        BLOCK_F_ARR.append(block_force_xy)
        FC_ARR.append(fc)

        #print('fc:',fc)
        if DEBUG:
            return np.hstack((state_dt,fc,))
        return state_dt

    def dynamics(self, t, state, u, DEBUG=False):
        """Returns the state derivative of the planar pusher model.
        Inputs:
            t: time 
            state: state vector [x, y, px, py, x_dot, y_dot, BC_body1, BC_body2,
            BC_body1_dot, BC_body2_dot] 
            u: control input [px_dot, py_dot]

        Outputs:
            state_dot: state derivative"""

        #print('t:',t)
        # Unpack state
        x, y, p_body1, p_body2, x_dot, y_dot = state
        if u is None:
            # if t < 2.5:
            #     px_dot = 0.05
            #     py_dot = 0
            # elif t >= 0.5:
            #     px_dot = -0.05
            #     py_dot = 0
            p_x_dot = 0.05
            p_y_dot = 0
        else:
            p_x_dot, p_y_dot = u
        #p_body1_dot = -p_x_dot*np.cos(p_body2) - p_y_dot*np.sin(p_body2) + np.linalg.norm(np.array([x_dot,y_dot])) # Need to consider the motion of the block. Check if the input is in xy or local coords. I think it should be local to allow control coherent koopman
        p_body1_dot = -p_x_dot*np.cos(p_body2) - p_y_dot*np.sin(p_body2) - (np.dot(np.array([x_dot,y_dot]),np.array([-np.cos(p_body2),-np.sin(p_body2)]))) # Need to consider the motion of the block. Check if the input is in xy or local coords. I think it should be local to allow control coherent koopman
        p_body2_dot = -(-p_x_dot*np.sin(p_body2) + p_y_dot*np.cos(p_body2))/p_body1 - (np.dot(np.array([x_dot,y_dot]),np.array([-np.sin(p_body2),-np.cos(p_body2)])))/p_body1 # Need to consider the motion of the block
        #p_body2_dot = -(-p_x_dot*np.sin(p_body2) + p_y_dot*np.cos(p_body2))/p_body1 # Need to consider the motion of the block
        pusher_body = np.array([p_body1, p_body2])
        
        #BC_body = np.array([BC_body1, BC_body2])       
        block_xy = np.array([x, y])

        p_body_dot = np.array([p_body1_dot, p_body2_dot])

        # Determine the forces from the pusher
        # These are not considered in the case of a massless contact
        # pusher_f = -self.pusher_k * (C_xy - pusher_xy)
        # pusher_f += -self.pusher_b * (C_xy_dot - pusher_body_xy)
        # pusher_f_body = R.T @ pusher_f
        # fx = pusher_f_body[0] # force from spring
        # fy = pusher_f_body[1] # force from spring

        # Determine the contact forces
        # Assume that the pusher is always aligned rotationally with the local block coordinates
        pen_dist_c1 = -(pusher_body[0] - (self.block_diameter/2))
        #print('pusher body 0:',pusher_body[0])
        #print('pen_dist_c1:',pen_dist_c1)
        fc = 0 # fc is in the body frame
        if pen_dist_c1 > 0:
            #fc += -self.contact_k * pen_dist
            #fc += -self.contact_b * p_body_dot[0]
            fc += -self.pusher_k * pen_dist_c1


        # Determine the limits on the forces from the pusher on the block
        # Note that in this case, we assume that there is no friction from the pusher contact with the object!
        # max_push_f = abs(self.pusher_mu * fc) # max limit from friction cone
        # max_push_f is equal to zero due to the no-friction assumption!
        # if abs(fy) > max_push_f:
        #     fy_block = (
        #         np.sign(-fy) * max_push_f
        #     )  # set the force experienced by the block to lie on or within the friction cone
        # else:
        #     fy_block = -fy
        fnorm_block = -fc # This would be entirelt
        ftan_block = 0 # No friction from the pusher
        m_block = np.cross(
           [pusher_body[0],0], [fnorm_block, ftan_block]
        )  # This might be covered by the motion cone. Check this later.

        block_force_body = np.array([fnorm_block, ftan_block, m_block])
        R2 = np.array([[np.cos(-pusher_body[1]), np.sin(-pusher_body[1])],[-np.sin(-pusher_body[1]), np.cos(-pusher_body[1])]])
        block_force_xy = np.array([0,0,m_block])
        block_force_xy[:2] = R2 @ block_force_body[:2]
        #block_force_xy[:2] =  * fx_block
        #print('block_force_xy:',block_force_xy)
        #print('block_force_body:',block_force_body)
        #print('theta:',theta)
        surface_f = -self.table_fric*np.array([x_dot, y_dot, 0])

        # Determine the net force on the block
        net_f_block = np.zeros((3))
        #net_f_block[:2] = (surface_f[:2] + block_force[:2]) @ body_to_xy
        net_f_block[:2] = surface_f[:2] + block_force_xy[:2]
        #net_f_block[:2] = (surface_f[:2] + block_force[:2]) @ body_to_xy + np.cross([0,0,theta_dot], [BC_xy[0], BC_xy[1], 0])[:2]
        net_f_block[2] = surface_f[2] + block_force_xy[2]
        if net_f_block[2] != 0:
            raise ValueError('Net force in rotational direction is not zero!')
        # Determine the net force on the end effector. Due to massless pusher, this doesn't matter
        #net_f_ee = np.array([fc,fy]) @ body_to_xy + pusher_f

        # # Determine the forces on the pusher due to the quasi-static assumption, it is assumed that
        # # the friction from the ground fully balances out the forces being applied by the pusher's
        # # end effector. This means that the pusher's velocities remain constant? EXCEPT when sliding
        # # tangential to the object?
        # ee_y_f = (
        #     fy - fy_block
        # )  # Should only be non-zero when lying on the edge of the motion cone
        # # The below allows the pusher end effector to move away from the object. This may
        # # dramatically increase the complexity of the system, and we should probably start out with
        # # the assumption that this cannot happen.
        # if fx < 0:
        #     ee_x_f = fx
        # else:
        #     ee_x_f = x_dot + np.cross([0,0,theta_dot], [ex, ey, 0])[2]

        # Determine the state derivatives

        x_dt = x_dot
        y_dt = y_dot
        px_dt = p_body_dot[0]
        py_dt = p_body_dot[1]
        #x_dot_dt = block_xdt[0]
        #y_dot_dt = block_xdt[1]
        #theta_dot_dt = block_xdt[2]
        x_dot_dt = net_f_block[0] / self.block_mass
        y_dot_dt = net_f_block[1] / self.block_mass
        #BC_xy_dot_dt = net_f_ee - np.array([x_dot_dt, y_dot_dt])
        #BC_body_dot_dt = BC_xy_dot_dt @ xy_to_body
        #BC_body1_dt = BC_body1_dot
        #BC_body2_dt = BC_body2_dot
        #ex_dot_dt = ee_x_f
        #ey_dot_dt = ee_y_f
        #BC_body1_dot_dt = BC_body_dot_dt[0]
        #BC_body2_dot_dt = BC_body_dot_dt[1]
        # if x_dt != 0:
        #     print('moving block')
        state_dt = np.array(
            [
                x_dt,
                y_dt,
                px_dt,
                py_dt,
                x_dot_dt,
                y_dot_dt
                #BC_body1_dt,
                #BC_body2_dt,
                #BC_body1_dot_dt,
                #BC_body2_dot_dt,
            ]
        )

        SURFACE_F_ARR.append(surface_f)
        BLOCK_F_ARR.append(block_force_xy)
        FC_ARR.append(fc)

        #print('fc:',fc)
        if DEBUG:
            return np.hstack((state_dt,fc,))
        return state_dt

    
    def plot_trajectory_graphs(self, state_array,dt=0.01,show=True):
        """Plots a trajectory of the planar pusher model as line graphs."""
        fig, axs = plt.subplots(2, 4)
        t = np.arange(0, len(state_array)*dt, dt)
        state_names = ['x', 'y', 'px', 'py', 'x_dot', 'y_dot']
        for i in range(2):
            for j in range(3):
                axs[i, j].plot(t, state_array[:, 3*i+j])
                axs[i, j].set_title(state_names[3*i+j])
        if show:
            plt.show()
    
    def animate_trajectory(self, state_array, dt=0.01,SAVE=False,str=None,lims=None):
        """Animates the trajectory of the planar pusher model.
        Inputs:
            state_array: an array of states over time
            dt: the time step between each state"""
        
        def init_ani():
            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)
            #ax.add_patch(ee_patch)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        pusher_patch = plt.Circle((0, 0), self.block_diameter/20, fc='r',zorder=2)
        block_patch = plt.Circle((0, 0), self.block_diameter/2, fc='b')
        anim = animation.FuncAnimation(fig, self.pusher_animation_func, init_func=init_ani(),frames=len(state_array), 
                                       fargs=(state_array, pusher_patch, block_patch))
        if lims is None:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        else:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
        if SAVE:
            writervideo = animation.FFMpegWriter(fps=10)
            anim.save(str+'.mp4', writer=writervideo)
        plt.show()

    def animate_trajectory_comparison(self, state_array, comparison_array, dt=0.01,SAVE=False,str=None,lims=None):
        """Animates the trajectory of the planar pusher model.
        Inputs:
            state_array: an array of states over time
            dt: the time step between each state"""
        
        def init_ani():
            ax.add_patch(pusher_patch)
            ax.add_patch(block_patch)
            #ax.add_patch(ee_patch)
            ax.add_patch(comparison_block_patch)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True) 
        pusher_patch = plt.Circle((0, 0), self.block_diameter/20, fc='r',zorder=2)
        pusher2_patch = plt.Circle((0, 0), self.block_diameter/20, fc='r',zorder=2)
        block_patch = plt.Circle((0, 0), self.block_diameter/2, fc='b')
        comparison_block_patch = plt.Circle((0,0), self.block_diameter/2, fc='y', alpha=0.5)
        anim = animation.FuncAnimation(fig, self.pusher_animation_func_comparison, init_func=init_ani(),frames=len(state_array), 
                                       fargs=(state_array, comparison_array, pusher_patch, block_patch, comparison_block_patch))
        if lims is None:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
            ax.set_aspect('equal', adjustable='box')
        if SAVE:
            writervideo = animation.FFMpegWriter(fps=10)
            anim.save(str+'.mp4', writer=writervideo)
        plt.show()
    
    def pusher_animation_func(self, frame, state_array, pusher_patch, block_patch):
        """Function to animate the pusher."""
        current_state = state_array[frame]
        
        block_patch.set_center([current_state[0],current_state[1]])   
        pusher_patch.set_center([current_state[2], current_state[3]])

    def pusher_animation_func_comparison(self, frame, state_array, comparison_array, pusher_patch, block_patch, comparison_block_patch):
        """Function to animate the pusher."""
        current_state = state_array[frame]
        
        block_patch.set_center([current_state[0],current_state[1]])   
        pusher_patch.set_center([current_state[2], current_state[3]])

        comparison_state = comparison_array[frame]
        comparison_block_patch.set_center([comparison_state[0], comparison_state[1]])
    
    def Convert_to_xy(self, state_arr):
        """Takes in a trajectory of states and converts them to global xy coordiantes.
        Inputs:
            state: the trajectory of states [x, y, theta, p_body1, p_body2, x_dot, y_dot, theta_dot, BC_body1, BC_body2, BC_body1_dot, BC_body2_dot]"""
        new_state_arr = []
        for state in state_arr:
            x, y, p_body1, p_body2, x_dot, y_dot = state
            px = -p_body1*np.cos(p_body2) + x
            py = -p_body1*np.sin(p_body2) + y
            p_xy = np.array([px, py])
            new_state = np.array([x, y, p_xy[0], p_xy[1], x_dot, y_dot])
            new_state_arr.append(new_state)
        assert len(new_state_arr) == len(state_arr)
        return new_state_arr
    
    # def Convert_to_body_vel(self, state_arr):
    #     """Takes in a trajectory of states and converts them to include the velocity in body coordinates.
    #     Inputs:
    #         state: the trajectory of states [x, y, theta, p_body1, p_body2, x_dot, y_dot, theta_dot]"""
    #     new_state_arr = []
    #     for state in state_arr:
    #         x, y, theta, p_body1, p_body2, x_dot, y_dot, theta_dot = state
    #         xy_to_body = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T
    #         body_to_xy = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T
    #         x_bodydot = np.array([x_dot, y_dot]) @ xy_to_body
    #         new_state = np.array([x, y, theta, p_body1, p_body2, x_bodydot[0], x_bodydot[1], theta_dot])
    #         new_state_arr.append(new_state)
    #     assert len(new_state_arr) == len(state_arr)
    #     return new_state_arr
    
    # def Convert_to_body_states(self, state_arr):
    #     """Takes in a trajectory of states and converts them to body coordinates.
    #         Inputs:
    #             state: the trajectory of states [x, y, theta, p_body1, p_body2, x_dot, y_dot, theta_dot, BC_body1, BC_body2, BC_body1_dot, BC_body2_dot]"""
    #     new_state_arr = []
    #     for state in state_arr:
    #         x, y, theta, p_body1, p_body2, x_dot, y_dot, theta_dot = state
    #         xy_to_body = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T
    #         body_to_xy = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T
    #         x_bodydot = np.array([x_dot, y_dot]) @ xy_to_body
    #         x_body = np.array([x, y]) @ xy_to_body
    #         new_state = np.array([x_body[0], x_body[1], theta, p_body1, p_body2, x_bodydot[0], x_bodydot[1], theta_dot])
    #         new_state_arr.append(new_state)
    #     assert len(new_state_arr) == len(state_arr)
    #     return new_state_arr
    
    # def Convert_from_body_states(self, state_arr):
    #     """Takes in a trajectory of states in body coordinates and converts them for simulation.
    #     Inputs:
    #         state: the trajectory of states [x_1, x_2, theta, p_body1, p_body2, x1_dot, x2_dot, theta_dot, BC_body1, BC_body2, BC_body1_dot, BC_body2_dot]"""
    #     new_state_arr = []
    #     for state in state_arr:
    #         x1, x2, theta, p_body1, p_body2, x1_dot, x2_dot, theta_dot = state
    #         xy_to_body = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T
    #         body_to_xy = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T
    #         xy_dot = np.array([x1_dot, x2_dot]) @ body_to_xy
    #         xy = np.array([x1, x2]) @ body_to_xy
    #         new_state = np.array([xy[0], xy[1], theta, p_body1, p_body2, xy_dot[0], xy_dot[1], theta_dot])
    #         new_state_arr.append(new_state)
    #     assert len(new_state_arr) == len(state_arr)
    #     return new_state_arr
    
    # def Convert_to_xy_vel(self, state_arr):
    #     """Takes in a trajectory of states and converts them to include the velocity in xy coordinates.
    #     Inputs:
    #         state: the trajectory of states [x, y, theta, p_body1, p_body2, cbody1_dot, cbody2_dot, theta_dot, BC_body1, BC_body2, BC_body1_dot, BC_body2_dot]"""
    #     new_state_arr = []
    #     for state in state_arr:
    #         x, y, theta, p_body1, p_body2, cbody1_dot, cbody2_dot, theta_dot = state
    #         xy_to_body = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T
    #         body_to_xy = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T
    #         xy_dot = np.array([cbody1_dot, cbody2_dot]) @ body_to_xy
    #         new_state = np.array([x, y, theta, p_body1, p_body2, xy_dot[0], xy_dot[1], theta_dot])
    #         new_state_arr.append(new_state)
    #     assert len(new_state_arr) == len(state_arr)
    #     return new_state_arr
                                 

def main(SAVE=False):
    """Main function which simulates an example of the planar pusher for testing."""
    # Define the parameters of the system
    block_diameter = 0.16
    block_mass = 0.827
    table_mu = 0.35
    #table_mu = 0.9
    pusher_mu = 0.3
    pusher_k = 1000

    # state = np.array(
    #     [0, 0, np.deg2rad(45), -block_diameter/2, 0, 0, 0, 0]
    # )  # [x, y, theta, px, py, x_dot, y_dot, theta_dot]

    # state = np.array(
    # [0, 0, np.deg2rad(0), -block_diameter/2, -0.9*block_diameter/2, 0, 0, 0]
    # )  # [x, y, theta, px, py, x_dot, y_dot, theta_dot]

    state = np.array(
        [0, 0, block_diameter/2, 0, 0, 0]
    )  # [x, y, px, py, x_dot, y_dot]

    # state = np.array(
    #     [0, 0, 0, -block_diameter/2, block_diameter/4, 0, 0, 0]
    # )  # [x, y, theta, px, py, x_dot, y_dot, theta_dot]
    pusher = Pusher(block_diameter, block_mass, table_mu, pusher_mu, pusher_k, state)
    p1_dot = -0.1
    p2_dot = 0
    x_dot = state[4]
    y_dot = state[5]
    p1 = state[2]
    p2 = state[3]
    px_dot = -p1_dot*np.cos(p2)+p1*np.sin(p2)*p2_dot
    py_dot = -p1_dot*np.sin(p2)-p1*np.cos(p2)*p2_dot
    #u = np.array([0.1, 0])
    u = np.array([px_dot, py_dot])
    #u = np.array([0,0])
    dt = 0.01
    #u = np.array([0,0.2])
    #u = None

    # Simulate the system
    #t = np.arange(0, 1, dt)
    t = np.arange(0,5,dt)
    # states = spi.solve_ivp(pusher.dynamics, [t[0], t[-1]], state, args=(None,), t_eval=t)
    states = spi.solve_ivp(pusher.dynamics, [t[0], t[-1]], state, args=(u,), t_eval=t)
    state_arr = states.y.T
    xy_state_arr = pusher.Convert_to_xy(list(state_arr))
    pusher.animate_trajectory(xy_state_arr, dt)
    xy_state_arr = np.array(xy_state_arr)

    # Plot the trajectory
    pusher.plot_trajectory_graphs(state_arr,dt)
    pusher.plot_trajectory_graphs(xy_state_arr,dt)

    # Save the trajectory
    if SAVE:
        np.save('pusher_trajectory_xy.npy', xy_state_arr)
        np.save('pusher_trajectory.npy', state_arr)
    

if __name__ == "__main__":
    DEBUG = False
    SAVE = True
    main(SAVE)
    SURFACE_F_ARR = np.array(SURFACE_F_ARR)
    BLOCK_F_ARR = np.array(BLOCK_F_ARR)
    FC_ARR = np.array(FC_ARR)
    # Plot SURFACE_F_ARR:
    if DEBUG:
        plt.figure()
        plt.plot(SURFACE_F_ARR[:,0])
        plt.plot(SURFACE_F_ARR[:,1])
        plt.plot(SURFACE_F_ARR[:,2])
        plt.show()
        plt.figure()
        plt.plot(BLOCK_F_ARR[:,0])
        plt.plot(BLOCK_F_ARR[:,1])
        plt.plot(BLOCK_F_ARR[:,2])
        plt.show()
        plt.figure()
        plt.plot(FC_ARR)
        plt.show()
