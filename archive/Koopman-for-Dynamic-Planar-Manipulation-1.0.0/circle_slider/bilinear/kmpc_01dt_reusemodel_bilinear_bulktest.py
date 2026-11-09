# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, sin, cos, log, pi
import math
import time
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from scipy import sparse
from scipy.integrate import odeint, solve_ivp
from scipy import interpolate
from scipy.interpolate import CubicSpline

#import spatialmath as sm
from planar_pusher import Pusher
import torch
import seaborn as sns
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from gurobi_ml import add_predictor_constr
from gurobi_ml.torch.sequential import add_sequential_constr
from nn_multistep_loss_1_bilinear.helpers.networkarch import NeuralNetwork

# %%
#T = 2
#T = 3 # 0.003 cost
#T = 4
#T = 5 # 0.017 cost
#T = 7
#T = 9 # 0.14 cost
#T = 7
T = 3
best_solution = None
NUMERICAL_ERROR_LIST = []

# class NeuralNetwork(torch.nn.Module):
#     def __init__(self, N_x = 1, N_h = 16, N_e = 10):
#         super(NeuralNetwork, self).__init__()
#         D_x = N_x
#         D_h = N_h
#         D_e = N_e
#         # Store dimensions in instance variables
#         self.D_x = D_x
#         self.D_e = D_e
#         D_xi = self.D_x + self.D_e

#         self.enc = nn.Sequential(
#             nn.Linear(D_x, D_h),
#             nn.ReLU(),
#             nn.Linear(D_h, D_h),
#             nn.ReLU(),
#             nn.Linear(D_h, D_h),
#             nn.ReLU(),
#             nn.Linear(D_h, D_h),
#             nn.ReLU(),
#             nn.Linear(D_h, D_h),
#             nn.ReLU(),
#             nn.Linear(D_h, D_h),
#             nn.ReLU(),
#             nn.Linear(D_h, D_h),
#             nn.ReLU(),
#             nn.Linear(D_h, D_e),
#         )
#         # Linear dynamic model matrices
#         self.A = torch.nn.Linear(D_xi, D_xi, bias=False)
#         self.B = torch.nn.Linear(2, D_xi, bias=False) # 2 is the dimension of the control input

#     def forward(self, x: torch.Tensor, ctrl: torch.Tensor):
#         '''Function to propagate the state and encoded state forward in time.
        
#         Args:
#             x (torch.Tensor): State of the system.
            
#         Returns:
#             x_tp1 (torch.Tensor): State of the system at the next time step.
#             eta_tp1 (torch.Tensor): Encoded state of the system at the next time step.'''
#         xs   = x
#         eta  = self.enc(xs)
#         xi = torch.cat((xs,eta), 1) # lines change
#         #xi = torch.cat((xs,eta), 0) # lines change

#         x_tp1, eta_tp1 = self.ldm(xi, ctrl)

#         return x_tp1, eta_tp1

#     def ldm(self, xi: torch.Tensor, ctrl: torch.Tensor):
#         '''Function to propagate the encoded state forward in time.
        
#         Args:
#             eta (torch.Tensor): Encoded state of the system.
            
#         Returns:
#             eta_tp1 (torch.Tensor): Encoded state of the system at the next time step.'''
#         xi_tp1 = self.A(xi) + self.B(ctrl)
#         #print('xi_tp1: ', xi_tp1.shape)
#         eta_tp1 = xi_tp1[:,self.D_x:]
#         x_tp1 = xi_tp1[:,:self.D_x]
#         return x_tp1, eta_tp1


def flatten(a):
    return np.array(a).flatten()

def calculate_xi(x, model,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),dtype=torch.float32):
    eta = model.enc(torch.tensor(x,device=device).type(dtype)).detach().cpu().numpy()
    if x.ndim == 1:
        xi = np.concatenate((x, eta))
    else:
        xi = np.concatenate((x, eta), axis=1)
    return xi

def calculate_xi_wpos(x, model,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),dtype=torch.float32):
    if x.ndim == 1:
        x_nopos = x[2:]
    else:
        x_nopos = x[:,2:]
    xi = calculate_xi(x_nopos, model,device=device,dtype=dtype)
    if x.ndim == 1:
        xi_wpos = np.concatenate((np.array([x[0],x[1]]), xi))
    else:
        xi_wpos = np.concatenate((x[:,0:2].reshape(-1,2), xi), axis=1) # Check reshape
    return xi_wpos

def simulation(x, ctrl, pusher, dt=0.01, params_list =None):
    x_init = x
    tspan = [0.0, dt]
    #z = odeint(ring_shaft.dynamics, x_init, tspan, args=(ctrl,))
    z = solve_ivp(pusher.dynamics, tspan,y0=x_init, args=(ctrl,), method='RK45', t_eval=tspan, rtol=1e-6, atol=1e-6)
    # x_init = z[-1]
    x_init = z.y.T[-1]
    return x_init

def check_dkn(model, xi, time_horizon, A, B, control=None):
    x = xi[:num_states+2]
    xi_partial = xi[2:]
    xi_arr = [xi]
    x_arr = [x]
    x_true_arr = [x]
    x_true = x
    for step in range(time_horizon):
        if control is None:
            control_step = np.array([0,0])
        else:
            control_step = control[step]
        control_step = control_step * 0.01
        xi_partial = A @ xi_partial + B @ control_step*dt
        xi_new = xi.copy()
        xi_new[2:] = xi_partial
        xi_new[0] = xi[0] + 0.1*(xi[4]+xi_new[4])/2 
        xi_new[1] = xi[1] + 0.1*(xi[5]+xi_new[5])/2 
        xi_arr.append(xi_new)
        x_arr.append(xi_new[:num_states+2])
        p1_dot = control_step[0]
        p2_dot = control_step[1]
        x_dot = xi[4]
        y_dot = xi[5]
        p1 = xi[2]
        p2 = xi[3]
        px_dot = -p1_dot*np.cos(p2)+p1*np.sin(p2)*p2_dot
        py_dot = -p1_dot*np.sin(p2)-p1*np.cos(p2)*p2_dot
        control_step = np.array([px_dot, py_dot])
        x_true = simulation(x_true, control_step, pusher, dt, model)
        xi_true = calculate_xi_wpos(x_true, model)
        #x_true = xi_true[:num_states]
        x_true_arr.append(x_true)
    return np.array(x_arr), np.array(x_true_arr)

def plot_trajectory(state_data, state_data2 = None, SAVE=False, save_name=''):
    fig, axs = plt.subplots(4,3, figsize=(15, 10))
    fig.suptitle('System States')
    for i in range(num_states+2):
        axs[i//3, i%3].plot(state_data[:,i])
        axs[i//3, i%3].set_title('State ' + str(i))
        if state_data2 is not None:
            axs[i//3, i%3].plot(state_data2[:,i], color='r', linestyle='--')
        # Plot x_ref as well
        #axs[i//3, i%3].plot(conduit_traj[:,i], color='r', linestyle='--')
    if SAVE:
        plt.savefig('plot_' + save_name + '.png')
    plt.show()

def mpc_gurobi_create_model(x0, x_ref, A, B, predictor, DEBUG=False,Q_base=np.eye(12),R_base=np.eye(6)):
    """
    Inputs:
        x0: Current state of the system with shape [1, lifted_dim]
        x_ref: Reference state of the system with shape [T, lifted_dim]
        A: linear state transformation
        B: linear control transformation
        Q_base: Base cost matrix for the states
        R_base: Base cost matrix for the control
        """
    # Both x0 and x_ref are in the body frame
    m = gp.Model("MPC")
    m.Params.DualReductions = 0
    N = T
    xmin = -5e37
    xmax = 5e37
    umin = -3
    umax = 3
    xmin = np.tile(xmin, (N+1,1))
    xmax = np.tile(xmax, (N+1,1))
    umin = np.tile(umin, (N,1))
    umax = np.tile(umax, (N,1))
    u2min = -50
    u2max = 50
    u2min = np.tile(u2min, (N,1))
    u2max = np.tile(u2max, (N,1))
    umin = np.concatenate((umin, u2min), axis=1)
    umax = np.concatenate((umax, u2max), axis=1)
    Q = np.eye(A.shape[0]+2)*0
    Q[0][0] = Q_base[0][0]
    Q[1][1] = Q_base[1][1]
    Q[0][1] = Q_base[0][1]
    Q[1][0] = Q_base[1][0]
    Q[:6,:6] = Q_base
    R = R_base
    obs_num = A.shape[0]
    
    x = m.addMVar(shape=(N+1,obs_num+2), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
    z = m.addMVar(shape=(N+1,obs_num+2), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z')
    u = m.addMVar(shape=(N,2), lb=umin, ub=umax, name='u')
    du = m.addMVar(shape=(1,2), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='du')
    
    x0_constr = m.addConstr(x[0, :] == x0.T, name='x0constr')
    z_constr_arr = []

    # Load the model (should match training config)
    model = NeuralNetwork(N_x=4, N_h=32, N_e=100) # Adjust N_x, N_h, N_e as needed
    checkpoint = torch.load('nn_multistep_loss_1_bilinear/model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_dict'])
    model.eval()
    for k in range(N):
        z_constr = m.addConstr(z[k, :] == x[k, :] - x_ref[k, :], name='zconstr_'+str(k))
        z_constr_arr.append(z_constr)
        # Use add_sequential_constr with B_bilinear
        B_out_constr = add_sequential_constr(m, model.B_bilinear, x[k, 2:], output_shape=(A.shape[1], 2))
        B_out = B_out_constr.output
        B_out = B_out.reshape((A.shape[1], 2))
        B_ctrl = m.addMVar(shape=(A.shape[1],), lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'B_ctrl_{k}')
        for i in range(A.shape[1]):
            m.addConstr(B_ctrl[i] == B_out[i,0]*u[k,0] + B_out[i,1]*u[k,1])
        m.addConstr(x[k+1, 2:] == A @ x[k, 2:] + B_ctrl*0.01*0.1, name='koop_pred_'+str(k+1))
        m.addConstr(x[k+1,0] == x[k,0] + 0.1*(x[k,4]+x[k+1,4])/2)
        m.addConstr(x[k+1,1] == x[k,1] + 0.1*(x[k,5]+x[k+1,5])/2)
    obj1 = sum(z[k, :] @ Q @ z[k, :] for k in range(1,N+1))
    obj2 = sum(u[k, :] @ R @ u[k, :] for k in range(N))
    R2 = np.eye(2) * 1e5
    #m.setObjective(obj1 + obj2, GRB.MINIMIZE)
    #m.params.NonConvex = 2
    # m.params.FeasibilityTol = 1e-2
    # m.params.OptimalityTol = 1e-2
    #m.params.BarConvTol = 1e-2
    #m.params.BarIterLimit = 4
    #m.params.BarQCPConvTol = 1e-2
    #m.params.TimeLimit = 0.01
    if DEBUG == False:
        m.Params.LogToConsole = 0
        m.Params.OutputFlag = 0
    m.optimize(my_callback)
    control = u.X[0:1,:]/100
    objval = m.objval

    return control, objval, 2, m, u, x0_constr, z_constr_arr

def my_callback(model, where):
    if where == GRB.Callback.MIPNODE:  # For MIP/QP nodes
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            # Retrieve current (potentially infeasible) solution
            node_vals = model.cbGetNodeRel(model.getVars())
            best_solution = node_vals  # Store it externally

    elif where == GRB.Callback.MIPSOL:  # When Gurobi finds a feasible solution
        best_solution = model.cbGetSolution(model.getVars())

def runtime_cutoff(m, where):
    runtime = m.cbGet(GRB.Callback.RUNTIME)
    if runtime > 0.003:
        m.terminate()

def mpc_gurobi_original(x0, x_ref, A, B, DEBUG=False,Q_base=np.eye(12),R_base=np.eye(6)):
    """
    Inputs:
        x0: Current state of the system with shape [1, lifted_dim]
        x_ref: Reference state of the system with shape [T, lifted_dim]
        A: linear state transformation
        B: linear control transformation
        Q_base: Base cost matrix for the states
        R_base: Base cost matrix for the control
        """
    # Both x0 and x_ref are in the body frame
    m = gp.Model("MPC")
    m.Params.DualReductions = 0
    N = T
    xmin = -5e37
    xmax = 5e37
    umin = -1
    umax = 1
    xmin = np.tile(xmin, (N+1,1))
    xmax = np.tile(xmax, (N+1,1))
    umin = np.tile(umin, (N,1))
    umax = np.tile(umax, (N,1))
    u2min = -50
    u2max = 50
    u2min = np.tile(u2min, (N,1))
    u2max = np.tile(u2max, (N,1))
    umin = np.concatenate((umin, u2min), axis=1)
    umax = np.concatenate((umax, u2max), axis=1)
    Q = np.eye(A.shape[0]+2)*0
    Q[0][0] = Q_base[0][0]
    Q[1][1] = Q_base[1][1]
    Q[0][1] = Q_base[0][1]
    Q[1][0] = Q_base[1][0]
    Q[:6,:6] = Q_base
    R = R_base
    obs_num = A.shape[0]
    
    x = m.addMVar(shape=(N+1,obs_num+2), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
    z = m.addMVar(shape=(N+1,obs_num+2), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z')
    u = m.addMVar(shape=(N,2), lb=umin, ub=umax, name='u')

    x0_constr = m.addConstr(x[0, :] == x0.T)
    z_constr_arr = []
    for k in range(N):
        # [x_1, x_2, theta, px, py, x1_dot, x2_dot, theta_dot]
        z_constr = m.addConstr(z[k, :] == x[k, :] - x_ref[k, :], name='zconstr_'+str(k))
        z_constr_arr.append(z_constr)
        m.addConstr(x[k+1, 2:] == A @ x[k, 2:] + B @ u[k,:]*0.01*0.1, name='koop_pred_'+str(k+1))
        m.addConstr(x[k+1,0] == x[k,0] + 0.1*(x[k,4]+x[k+1,4])/2) # Calculate x_1
        m.addConstr(x[k+1,1] == x[k,1] + 0.1*(x[k,5]+x[k+1,5])/2) # Calculate x_2
        m.addConstr(x[k+1,2] <= 0.1)

    # # Experiment with setting a constraint on the vertical height of the pusher
    # Contraint on the block's state
    m.addConstr(x[N,2] <= 0.1)

    z_constr = m.addConstr(z[N, :] == x[N, :] - x_ref[N, :],name='zconstr_'+str(N))
    z_constr_arr.append(z_constr)
    obj1 = sum(z[k, :] @ Q @ z[k, :] for k in range(1,N+1))
    obj2 = sum(u[k, :] @ R @ u[k, :] for k in range(N))
    m.setObjective(obj1 + obj2, GRB.MINIMIZE)
    #m.params.NonConvex = 2
    if DEBUG == False:
        m.Params.LogToConsole = 0
        m.Params.OutputFlag = 0
    m.optimize()
    #print("Runtime:",m.Runtime)
    if m.Runtime > 0.1:
        print("Runtime:",m.Runtime)

    opt_status = m.status
    #print('Optimization status:',opt_status)
    RELAX = False
    if opt_status == 3:
        if RELAX:
            print('Infeasible, relaxing...')
            orignumvars = m.NumVars
            m.feasRelaxS(1, False, False, True)
            m.optimize()
            opt_status = m.status
            print('Optimization status:',opt_status)
            if opt_status == 3:
                print('Infeasible')
                orignumvars = m.NumVars
                m.feasRelaxS(1, False, False, True)
                m.optimize()
                opt_status = m.status
                print('Optimization status:',opt_status)
            elif opt_status == 12:
                print("Numerical error")
                control = np.array([[0,0]])
                objval = 0
            control = u.X/100
            objval = m.objval
            # print the values of the artificial variables of the relaxation
            print('\nSlack values:')
            slacks = m.getVars()[orignumvars:]
            for sv in slacks:
                if sv.X > 0:
                    print('%s = %g' % (sv.VarName, sv.X))
                # if sv.X > 1e-9:
                #     print('%s = %g' % (sv.VarName, sv.X))
        else:
            print("Infeasible")
            control = np.array([[0,0]])
            objval = 0
    elif opt_status == 12:
        print("Numerical error")
        control = np.array([[0,0]])
        objval = 0
    else:
        control = u.X[0:1,:]/100
        objval = m.objval
    #control = np.array([[control[0][1],control[0][0]]])
    return control, objval, opt_status, m, u, x0_constr, z_constr_arr

def mpc_gurobi(x0, x_ref, m, u, x0_constr, z_constr_arr):
    """
    Inputs:
        x0: Current state of the system with shape [1, lifted_dim]
        x_ref: Reference state of the system with shape [T, lifted_dim]
        A: linear state transformation
        B: linear control transformation
        Q_base: Base cost matrix for the states
        R_base: Base cost matrix for the control
        """
    # Both x0 and x_ref are in the body frame
    N = T
    m.update()
    x0_constr.rhs = x0.T
    for ind in range(len(z_constr_arr)):
        z_constr = z_constr_arr[ind]
        z_constr.rhs = -x_ref[ind]
    m.optimize(my_callback)
    #u = m.getVars()[m.NumVars-N*2:m.NumVars]
    #print("Runtime:",m.Runtime)

    opt_status = m.status
    #print('Optimization status:',opt_status)
    RELAX = False
    if opt_status == 3:
        print("Infeasible")
        control = np.array([[0,0]])
        objval = 0
    elif opt_status == 12:
        print("Numerical error")
        control = np.array([[0,0]])
        #control = np.array([[umax[0][0]/100]])
        objval = 0
    else:
        control = u.X[0:1,:]/100
        objval = m.objval
    #control = np.array([[control[0][1],control[0][0]]])
    return control, objval, opt_status, m

def rotation_transform(state, ref_traj, theta, center):
    """
    Rotate the provided points by an angle theta around center. Return the points.
    Inputs:
        state np.array - [x,y,theta,p1,p2,x_dot,y_dot] state of the system
        ref_traj np.array - [[x0,y0],[x1,y1],...] array of points which define a reference trajectory
        theta float - radian angle of rotation
        center np.array - [x,y] point about which rotation is to occur
    Outputs:
        rotated_state, rotated_reference_array
    """
    
    def rotate_point(point, angle, center_point):
        # Translate point to origin
        translated_point = point - center_point
        # Rotate point
        rotated_point = np.array([
            translated_point[0] * np.cos(angle) - translated_point[1] * np.sin(angle),
            translated_point[0] * np.sin(angle) + translated_point[1] * np.cos(angle)
        ])
        # Translate point back
        return rotated_point + center_point
    
    def rotate_vector(vec, angle):
        return np.array([vec[0] * np.cos(angle) - vec[1] * np.sin(angle), vec[0] * np.sin(angle) + vec[1] * np.cos(angle)])

    x, y, p1, p2, x_dot, y_dot = state
    p2_rot = state[3] + theta
    #xdot_rot = rotate_point([x_dot,y_dot], theta, center)
    xdot_rot = rotate_vector([x_dot,y_dot], theta)
    ref_xy_array_rot = np.array([rotate_point(ref_xy, theta, center) for ref_xy in ref_traj]) # For now only considers xy of reference trajectory
    
    return np.array([x,y,p1,p2_rot,xdot_rot[0],xdot_rot[1]]), ref_xy_array_rot

def main_script(mu_t=10., SAVE=False, PLOT=False, init_config = None):
    dir = 'results/bulktest_fric'
    kmpc_run = True
    num_states = 4
    dt = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    max_timesteps = 400
    # Set initial values
    mu_p = 0.3
    init_err_x = 0.0
    mass_block = 0.156
    block_length = 0.16
    pusher_k = 1000
    max_energy = None
    init_state = np.array(
        [0, 0, block_length/2, np.deg2rad(0), 0, 0]) # Placeholder, x and y init redefined later
    conduit_traj = np.zeros((max_timesteps,num_states+2))
    conduit_traj[:,0] = 400*0.005*dt + init_state[0]
    conduit_traj[:,1] = conduit_traj[:,1] + init_state[1]
    ONNX_B_PATH = "B_bilinear.onnx"

    # Load the model
    num_obs = 100
    model = NeuralNetwork(N_x = num_states, N_e = num_obs, N_h = 32)
    checkpoint_stab = torch.load('nn_multistep_loss_1_bilinear/model.pt')
    model.load_state_dict(checkpoint_stab['model_dict'])
    model.to(device)
    init_state = init_config.copy()
    pusher = Pusher(block_length, mass_block, mu_t, mu_p, pusher_k, init_state)
    

    # Extract the A matrix from the model
    A = model.A.weight.detach().cpu().numpy()
    B_diag = np.diag(np.array([1,1]))
    B = np.zeros((2,A.shape[1]))
    B[:,0:2] = B_diag
    B = B.T
    #B = model.B.weight.detach().cpu().numpy()
    
    x_init = init_state
    x_ref = x_init

    xi = calculate_xi_wpos(x_init, model)
    xi_ref = calculate_xi_wpos(x_ref, model)

    Q_overall = np.eye(num_states+3)*1e-10
    Q_overall[:2,:2] = np.eye(2)*1e20
    Q_overall[2,2] = 1e10
    R_overall = np.eye(2)*dt*1e5

    Q_overall = np.eye(num_states+2)*1e-10
    Q_overall = np.eye(num_states+2)*0
    Q_overall[1,1] = 1e5
    Q_overall[0,0] = 1e5
    R_overall = np.eye(2)*dt*1e-5

    # Q2 set: This is a bit of a middle ground between Q1 and Q3
    Q = np.eye(num_states+2)*1e-10 # 0.003 cost total
    Q = np.eye(num_states+2)*0 # 0.003 cost total
    Q[0,0] = 1e1
    Q[1,1] = 1e1
    R = np.eye(2)*dt*1e-5
    R[1,1] = dt*1e-10

    state_data = []
    state_data.append(x_init)
    xi_data = []
    xi_data.append(xi)

    MPC_cost_arr = []
    cost_arr = []
    res_arr = []
    x_cost_arr = []
    y_cost_arr = []
    theta_cost_arr = []

    kmpc_times = []
    python1_times = []
    python2_times = []
    gurobi_times = []
    print('Starting controller...') 
    range_val = conduit_traj.shape[0]
    xi_nospeed = xi.copy()
    for step in range(max_timesteps-1):
        print("Step:",step)
        conduit_traj_ref = conduit_traj[step,:]
        if step <  range_val-T:
            conduit_traj_ref_arr = conduit_traj[step:step+T+1,:]
            # Convert the velocities to be in the body frame
            xi_ref = calculate_xi_wpos(conduit_traj_ref_arr, model)

        else:
            conduit_traj_ref_arr = conduit_traj_ref*np.ones((T+1,num_states+2))
            # Convert the velocities to be in the body frame
            xi_ref = calculate_xi_wpos(conduit_traj_ref_arr, model)

        HIGH_LEVEL_CTRL = False
        if kmpc_run:
            kmpc_start_time = time.time()
            python1_start_time = time.time()
            x_dot = xi[4]
            y_dot = xi[5]
            p1 = xi[2]
            p2 = xi[3]
            # Rotate the system so that the pusher is at 0 degs
            
            state_rot, ref_rot_xy = rotation_transform(xi[:6], xi_ref[:,:2],-p2, xi[:2])
            ref_rot = conduit_traj_ref_arr.copy()
            ref_rot[:,:2] = ref_rot_xy
            xi_rot = calculate_xi_wpos(state_rot, model)
            xi_rot_ref = calculate_xi_wpos(ref_rot, model)
            if step == 0:
                control, objval, opt_status, m, u, x0_constr, z_constr_arr = mpc_gurobi_create_model(xi_rot, xi_rot_ref, A=A, B=B, predictor=ONNX_B_PATH, DEBUG=False, Q_base=Q, R_base=R)
                control_prev = control
            python1_end_time = time.time()
            gurobi_start_time = time.time()
            dist_to_goal = np.linalg.norm(xi_rot[:2] - xi_rot_ref[0,:2])
            thresh2goal = 0
            if dist_to_goal > thresh2goal:
                #control, objval, opt_status, m, u, x0_constr, z_constr_arr = mpc_gurobi_create_model(xi_rot, xi_rot_ref,predictor, A=A,B=B, DEBUG=False,Q_base=Q,R_base=R)
                control, objval, opt_status, m = mpc_gurobi(xi_rot, xi_rot_ref, m, u, x0_constr, z_constr_arr)
            else:
                control = [[0,0]]
                objval = 0
                opt_status = 2
            gurobi_end_time = time.time()
            python2_start_time = time.time()

            control = control[0]
            control_old = control

            # CHECK THE FOLLOWING CONVERSION FOR CONTROL
            p1_dot = control[0]
            p2_dot = control[1]
            px_dot = -p1_dot*np.cos(p2)+p1*np.sin(p2)*p2_dot
            py_dot = -p1_dot*np.sin(p2)-p1*np.cos(p2)*p2_dot
            control = np.array([px_dot, py_dot])
            python2_end_time = time.time()
            kmpc_end_time = time.time()
            if opt_status == 12:
                NUMERICAL_ERROR_LIST.append(step)
            if step != 0:
                kmpc_times.append(kmpc_end_time-kmpc_start_time)
                python1_times.append(python1_end_time-python1_start_time)
                python2_times.append(python2_end_time-python2_start_time)
                gurobi_times.append(gurobi_end_time-gurobi_start_time)
 
        
        x= xi[:num_states+2]
        x_next = simulation(x, control, pusher, dt, model)
        state_data.append(x_next)
        xi = calculate_xi_wpos(x_next, model)
        xi_data.append(xi)
        z = x_next - xi_ref[0,:num_states+2]
        MPC_cost = z @ Q_overall @ z.T + control @ R_overall @ control.T
        x_cost_arr.append(z[0] * Q_overall[0,0] * z[0])
        y_cost_arr.append(z[1] * Q_overall[1,1] * z[1])
        theta_cost_arr.append(z[2] * Q_overall[2,2] * z[2])
        MPC_cost_arr.append(MPC_cost)
        cost = z[0]**2 + z[1]**2
        cost_arr.append(cost)
        res = [xi[:num_states+2], control_old]
        res_arr.append(res)
        xi_nospeed = xi.copy()
        xi_nospeed[5:8] = 0
        #xi = xi_nospeed.copy()

    print('Average KMPC time:', np.mean(kmpc_times))
    print('Max KMPC time:', np.max(kmpc_times))
    print('Total Cost:', np.sum(cost_arr))
    print('MPC Cost:',np.sum(MPC_cost_arr))
    print('How many numerical errors?', len(NUMERICAL_ERROR_LIST))

    #pusher.animate_trajectory_comparison(pusher.Convert_to_xy(list(state_data)), conduit_traj, dt, SAVE=True,str=dir,lims=[-0.1,0.5,-0.1,0.1])

    # Plot the control inputs
    control_data = np.array([res[1] for res in res_arr])
    if PLOT:
        fig, axs = plt.subplots(1,2, figsize=(15, 10))
        fig.suptitle('Control Inputs')
        for i in range(2):
            axs[i%3].plot(control_data[:,i])
            axs[i%3].set_title('Control ' + str(i))
            axs[i%3].set_ylim([-1,1])
        #plt.show()
        if SAVE:
            plt.savefig(dir+'control_plot.png')
        plt.show()

    # Plot the system's trajectory of the 12 states in subplots
    state_data = np.array(state_data)
    #xi_data = np.array(xi_data)
    #res_arr = np.array(res_arr)
    if PLOT:
        fig, axs = plt.subplots(4,3, figsize=(15, 10))
        fig.suptitle('System States')
        for i in range(num_states+2):
            axs[i//3, i%3].plot(state_data[:,i])
            axs[i//3, i%3].set_title('State ' + str(i))
            # Plot x_ref as well
            axs[i//3, i%3].plot(conduit_traj[:,i], color='r', linestyle='--')
        #plt.show()
        if SAVE:
            plt.savefig(dir+'state_plot.png')
        plt.show()

    # Plot the system's trajectory of the 12 states in subplots
    state_data_xy = pusher.Convert_to_xy(state_data)
    state_data_xy = np.array(state_data_xy)
    if PLOT:
        fig, axs = plt.subplots(4,3, figsize=(15, 10))
        fig.suptitle('System States')
        for i in range(num_states+2):
            axs[i//3, i%3].plot(state_data_xy[:,i])
            axs[i//3, i%3].set_title('State ' + str(i))
            # Plot x_ref as well
            axs[i//3, i%3].plot(conduit_traj[:,i], color='r', linestyle='--')
        if SAVE:
            plt.savefig(dir+'state_plotxy.png')
        plt.show()

    if SAVE:
        np.save(dir+'state_data.npy', state_data)
        np.save(dir+'control_data.npy', control_data)
        np.save(dir+'poserrsq_data.npy', cost_arr)
        np.save(dir+'MPC_cost_data.npy', MPC_cost_arr)
        results_file = dir+'results.txt'
        with open(results_file, 'w') as f:
            f.write('Average KMPC time: ' + str(np.mean(kmpc_times)) + '\n')
            f.write('Squared Position Error: ' + str(np.sum(cost_arr)) + '\n')
            f.write('MPC Cost: ' + str(np.sum(MPC_cost_arr)) + '\n')
    
    # Plot the cost
    if PLOT:
        fig, axs = plt.subplots(1,2, figsize=(15, 10))
        fig.suptitle('Costs')
        axs[0].plot(cost_arr)
        axs[0].set_title('Position Error Squared')
        axs[1].plot(MPC_cost_arr)
        axs[1].set_title('MPC Cost')
        if SAVE:
            plt.savefig(dir+'cost_plot.png')
        plt.show()

        # Plot the x,y,theta costs as separate lines on the same plot
        plt.figure()
        plt.plot(x_cost_arr, label='x')
        plt.plot(y_cost_arr, label='y')
        plt.plot(theta_cost_arr, label='theta')
        plt.legend()
        plt.title('Costs')
        if SAVE:
            plt.savefig(dir+'cost_plot_sep.png')
        plt.show()

        # Plot the kmpc times on a seaborn violinplot
        plt.figure()
        sns.stripplot([kmpc_times,python1_times,python2_times,gurobi_times])
        plt.title('KMPC Times')
        plt.ylabel('Time (s)')
        plt.legend(['KMPC','Python1','Python2','Gurobi'])
        plt.show()
        print("Done.")
    results = {'state_data': state_data, 'state_data_xy': state_data_xy, 'control_data': control_data, 'poserrsq_data': cost_arr, 'MPC_cost_data': MPC_cost_arr, 'kmpc_times': kmpc_times, 'mu_t' : mu_t, 'init_config': init_config}
    return results

if __name__ == '__main__':
    test_yerr_arr = np.array([0, 0.04550214,  0.03983901,  0.04573661, -0.00735578,  0.01862293,
        -0.04254677, -0.01577152,  0.03632274,  0.00784773,  0.02451669,
            0.00073406,  0.03805197,  0.01118484, -0.00390365,  0.04599985])
    test_p2 = np.array([0,np.deg2rad(180),np.deg2rad(-90)])
    results_arr = []
    test_mu = np.array([0.05,0.08,0.1,0.2,0.5,1.,10.])
    test_mu = [.35]
    for mu_count, mu_t in enumerate(test_mu):
        for init_count, test_yerr in enumerate(test_yerr_arr):
            for p2_count, test_p2_val in enumerate(test_p2):
                # test_yerr = 0
                # mu_t = 10.
                # test_p2_val = np.deg2rad(180)
                init_config = np.array([0, test_yerr, 0.16/2, test_p2_val, 0, 0])
                results = main_script(mu_t=mu_t, SAVE=False, PLOT=False, init_config=init_config)
                results_arr.append(results)
    # Save the results
    np.save('results/bilinear_results_035.npy', results_arr)
    print("All tests completed. Results saved to 'results/og_fric_results.npy'.")