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

# %%
T = 3
NUMERICAL_ERROR_LIST = []

class NeuralNetwork(torch.nn.Module):
    def __init__(self, N_x = 1, N_h = 16, N_e = 10):
        super(NeuralNetwork, self).__init__()
        D_x = N_x
        D_h = N_h
        D_e = N_e
        # Store dimensions in instance variables
        self.D_x = D_x
        self.D_e = D_e
        D_xi = self.D_x + self.D_e

        self.enc = nn.Sequential(
            nn.Linear(D_x, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_e),
        )
        # Linear dynamic model matrices
        self.A = torch.nn.Linear(D_xi, D_xi, bias=False)

    def forward(self, x: torch.Tensor):
        '''Function to propagate the state and encoded state forward in time.
        
        Args:
            x (torch.Tensor): State of the system.
            
        Returns:
            x_tp1 (torch.Tensor): State of the system at the next time step.
            eta_tp1 (torch.Tensor): Encoded state of the system at the next time step.'''
        xs   = x
        eta  = self.enc(xs)
        xi = torch.cat((xs,eta), 1) # lines change
        #xi = torch.cat((xs,eta), 0) # lines change

        x_tp1, eta_tp1 = self.ldm(xi)

        return x_tp1, eta_tp1

    def ldm(self, xi: torch.Tensor):
        '''Function to propagate the encoded state forward in time.
        
        Args:
            eta (torch.Tensor): Encoded state of the system.
            
        Returns:
            eta_tp1 (torch.Tensor): Encoded state of the system at the next time step.'''
        xi_tp1 = self.A(xi)
        #print('xi_tp1: ', xi_tp1.shape)
        eta_tp1 = xi_tp1[:,self.D_x:]
        x_tp1 = xi_tp1[:,:self.D_x]
        return x_tp1, eta_tp1

# def get_model_matrix():
#     Ad = pd.read_csv('./dmdmodels/qr' + '_sinecos_indic' + '.csv', header=None)
#     Ad = np.array(Ad.values)
#     Ad = sparse.csc_matrix(Ad)
#     Bd = np.load('dmdmodels/B_sinecos_indic.npy')
#     Bd = np.transpose(Bd)
#     return Ad, Bd

def RBF(center, x, kinds="gauss"):
    """
    This function returns RBF functions based on kmeans++ center positions and the states.
    """
    c = center
    epsilon = 0.1 # This is the hyperprameter!!!
    r = np.linalg.norm(x - c, axis=1, ord=2)
    if kinds == "gauss":
        return np.exp(-(r/(epsilon))**2)
    elif kinds == "quad":
        return 1/(1+(r/epsilon)**2)
    elif kinds == "inv_quadric":
        return 1/np.sqrt(1+(epsilon*r)**2)
    elif kinds == "thin":
        return r**2 * np.log(r)
    elif kinds == "quintic":
        return r**5
    elif kinds == "bump":
        func = np.exp(-1/(1-(epsilon*r)**2))
        # If >1/epliron, the values should be zero.
        return func
    else:
        pass

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
    x = xi[:num_states]
    xi_arr = [xi]
    x_arr = [x]
    x_true_arr = [x]
    x_true = x
    for step in range(time_horizon):
        if control is None:
            control_step = np.array([0,0])
        else:
            control_step = control[step]
        xi = A @ xi + B @ control_step*0.01
        xi_arr.append(xi)
        x_arr.append(xi[:num_states])
        x_true = simulation(x_true, control_step, pusher, dt, model)
        xi_true = calculate_xi_wpos(x_true, model)
        #x_true = xi_true[:num_states]
        x_true_arr.append(x_true)
    return x_arr, x_true_arr

def plot_trajectory(state_data, state_data2 = None, SAVE=False, save_name=''):
    fig, axs = plt.subplots(4,3, figsize=(15, 10))
    fig.suptitle('System States')
    for i in range(num_states):
        axs[i//3, i%3].plot(state_data[:,i])
        axs[i//3, i%3].set_title('State ' + str(i))
        if state_data2 is not None:
            axs[i//3, i%3].plot(state_data2[:,i], color='r', linestyle='--')
        # Plot x_ref as well
        #axs[i//3, i%3].plot(conduit_traj[:,i], color='r', linestyle='--')
    if SAVE:
        plt.savefig('plot_' + save_name + '.png')
    plt.show()

def mpc_gurobi_create_model(x0, x_ref, A, B, DEBUG=False,Q_base=np.eye(12),R_base=np.eye(6)):
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
    umin = -25
    umax = 25
    xmin = np.tile(xmin, (N+1,1))
    xmax = np.tile(xmax, (N+1,1))
    umin = np.tile(umin, (N,1))
    umax = np.tile(umax, (N,1))
    # umin = np.concatenate((umin, umin), axis=1)
    # umax = np.concatenate((umax, umax), axis=1)
    Q = np.eye(A.shape[0]+2)*0
    Q[0][0] = Q_base[0][0]
    Q[1][1] = Q_base[1][1]
    Q[0][1] = Q_base[0][1]
    Q[1][0] = Q_base[1][0]
    Q[:8,:8] = Q_base
    R = R_base
    obs_num = A.shape[0]
    
    x = m.addMVar(shape=(N+1,obs_num+2), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
    z = m.addMVar(shape=(N+1,obs_num+2), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z')
    u = m.addMVar(shape=(N,2), lb=umin, ub=umax, name='u')

    x0_constr = m.addConstr(x[0, :] == x0.T, name='x0constr')
    z_constr_arr = []
    for k in range(N):
        # [x_1, x_2, theta, px, py, x1_dot, x2_dot, theta_dot]
        z_constr = m.addConstr(z[k, :] == x[k, :] - x_ref[k, :], name='zconstr_'+str(k))
        z_constr_arr.append(z_constr)
        m.addConstr(x[k+1, 2:] == A @ x[k, 2:] + B @ u[k,:]*0.01*0.1, name='koop_pred_'+str(k+1))
        m.addConstr(x[k+1,0] == x[k,0] + 0.1*(x[k,5]+x[k+1,5])/2) # Calculate x_1
        m.addConstr(x[k+1,1] == x[k,1] + 0.1*(x[k,6]+x[k+1,6])/2) # Calculate x_2
        # # Contraint on the block's state
        m.addConstr(x[k+1,2]<=np.deg2rad(40))
        m.addConstr(x[k+1,2]>=-np.deg2rad(40))
        # Constraint on the x penetration of the pusher
        m.addConstr(x[k+1,4]<=0.91*pusher.block_length/2)
        m.addConstr(x[k+1,4]>=-0.91*pusher.block_length/2)
        #m.addConstr(u[k,1] == -2) THIS LEADS TO WEIRD RESULTS...I SUSPECT A VISUALIZATION ISSUE, AND IT'S ACTUALLY FINE?
    # # Experiment with setting a constraint on the vertical height of the pusher
    # Contraint on the block's state
    m.addConstr(x[N,2]<=np.deg2rad(80))
    m.addConstr(x[N,2]>=-np.deg2rad(80))
    m.addConstr(x[N,4]<=0.91*pusher.block_length/2)
    m.addConstr(x[N,4]>=-0.91*pusher.block_length/2)
    z_constr = m.addConstr(z[N, :] == x[N, :] - x_ref[N, :],name='zconstr_'+str(N))
    z_constr_arr.append(z_constr)
    
    obj1 = sum(z[k, :] @ Q @ z[k, :] for k in range(1,N+1))
    obj2 = sum(u[k, :] @ R @ u[k, :] for k in range(N))
    m.setObjective(obj1 + obj2, GRB.MINIMIZE)
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
    m.optimize()
    control = u.X[0:1,:]/100
    objval = m.objval

    return control, x.X, objval, 2, m, u, x, x0_constr, z_constr_arr

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

def mpc_gurobi(x0, x_ref, m, u, x_var, x0_constr, z_constr_arr):
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
        xi_pred = None
    elif opt_status == 12:
        print("Numerical error")
        control = np.array([[0,0]])
        #control = np.array([[umax[0][0]/100]])
        objval = 0
    else:
        control = u.X[0:1,:]/100
        objval = m.objval
        xi_pred = x_var.X
    #control = np.array([[control[0][1],control[0][0]]])
    return control, xi_pred, objval, opt_status, m


def mpc_gurobi_original(x0, x_ref, A, B, DEBUG=False,Q_base=np.eye(12),R_base=np.eye(6),env=None):
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
    m = gp.Model("MPC",env=env)
    m.Params.DualReductions = 0
    #A, B = get_model_matrix()
    #print(A)
    #print(B)
    N = T
    #print('N:',N)
    xmin = -5e37
    xmax = 5e37
    #umin = -0.5
    #umax = 0.5
    #umin = -100
    #umax = 100
    umin = -10
    umax = 10
    umin = -25
    umax = 25
    xmin = np.tile(xmin, (N+1,1))
    xmax = np.tile(xmax, (N+1,1))
    umin = np.tile(umin, (N,1))
    umax = np.tile(umax, (N,1))
    Q = np.eye(A.shape[0]+2)*0
    Q[0][0] = Q_base[0][0]
    #Q[0][0] = 1
    Q[1][1] = Q_base[1][1]
    Q[0][1] = Q_base[0][1]
    Q[1][0] = Q_base[1][0]
    Q[:8,:8] = Q_base
    #Q[1][1] = 0.9e2
    #Q[1][1] = 1e-4
    #Q[-1][-1] = 0.1
    #R = np.eye(1)*0.9e4*0.1
    #R = np.eye(1)*0.01
    R = R_base
    obs_num = A.shape[0]
    
    x = m.addMVar(shape=(N+1,obs_num+2), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
    z = m.addMVar(shape=(N+1,obs_num+2), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z')
    u = m.addMVar(shape=(N,2), lb=umin, ub=umax, name='u')

    m.addConstr(x[0, :] == x0.T)
    for k in range(N):
        # [x_1, x_2, theta, px, py, x1_dot, x2_dot, theta_dot]
        m.addConstr(z[k, :] == x[k, :] - x_ref[k, :], name='zconstr_'+str(k))
        m.addConstr(x[k+1, 2:] == A @ x[k, 2:] + B @ u[k,:]*0.01*0.1, name='koop_pred_'+str(k+1))
        #m.addConstr(x[k+1,1] == x[k,1]+0.01*(x[k,7]+x[k+1,7])/2)
        #m.addConstr(x[k+1,2] == x[k,2]+0.01*(x[k,7]+x[k+1,7])/2, name='theta_'+str(k+1)) # Calculate theta
        # m.addConstr(x_xyvel[k+1,5] == x[k+1,5] - x[k+1,6]*(x[k,2]+x[k+1,2])/2,name='xdot_'+str(k+1)) # Calculate x_dot
        # m.addConstr(x_xyvel[k+1,6] == x[k+1,6] + x[k+1,5]*(x[k,2]+x[k+1,2])/2,name='ydot_'+str(k+1)) # Calculate y_dot
        # m.addConstr(x[k+1,0] == x[k,0]+0.01*(x_xyvel[k+1,5]+x_xyvel[k,5])/2,name='x_'+str(k+1)) # Calculate x
        # m.addConstr(x[k+1,1] == x[k,1]+0.01*(x_xyvel[k+1,6]+x_xyvel[k,6])/2,name='x_'+str(k+1)) # Calculate y
        m.addConstr(x[k+1,0] == x[k,0] + 0.1*(x[k,5]+x[k+1,5])/2) # Calculate x_1
        m.addConstr(x[k+1,1] == x[k,1] + 0.1*(x[k,6]+x[k+1,6])/2) # Calculate x_2
        # Experiment with setting a constraint on the vertical height of the pusher
        # m.addConstr(x[k+1,4]<=0.91*pusher.block_length/2)
        # m.addConstr(x[k+1,4]>=-0.91*pusher.block_length/2)
        # # Contraint on the block's state
        m.addConstr(x[k+1,2]<=np.deg2rad(40))
        m.addConstr(x[k+1,2]>=-np.deg2rad(40))
        # Constraint on the x penetration of the pusher
        #m.addConstr(x[k+1,3]>= -0.06)
        #m.addConstr(x[k+1,3]>= -0.15)
        m.addConstr(x[k+1,4]<=0.91*pusher.block_length/2)
        m.addConstr(x[k+1,4]>=-0.91*pusher.block_length/2)
        #m.addConstr(u[k,1] == -2) THIS LEADS TO WEIRD RESULTS...I SUSPECT A VISUALIZATION ISSUE, AND IT'S ACTUALLY FINE?
    # # Experiment with setting a constraint on the vertical height of the pusher
    # Contraint on the block's state
    #m.addConstr(u[0,0]==10)
    # m.addConstr(x[N,2]<=np.deg2rad(40))
    # m.addConstr(x[N,2]>=-np.deg2rad(40))
    m.addConstr(x[N,2]<=np.deg2rad(80))
    m.addConstr(x[N,2]>=-np.deg2rad(80))
    m.addConstr(x[N,4]<=0.91*pusher.block_length/2)
    m.addConstr(x[N,4]>=-0.91*pusher.block_length/2)
    # m.addConstr(x[1,4]<=0.91*pusher.block_length/2)
    # m.addConstr(x[1,4]>=-0.91*pusher.block_length/2)
    # # Contraint on the block's state
    # m.addConstr(x[1,2]<=np.deg2rad(40))
    # m.addConstr(x[1,2]>=-np.deg2rad(40))
    # # Constraint on the x penetration of the pusher
    #m.addConstr(x[N-1,2]>= -0.06)
    #m.addConstr(x[1,2]>= -0.06)
    # m.addConstr(x[1,3]<=-0.04)
    # #m.addConstr(x[1,3]>=-0.0905)
    # #m.addConstr(x[1,3]<=-0.079)

    m.addConstr(z[N, :] == x[N, :] - x_ref[N, :],name='zconstr_'+str(N))
    
    obj1 = sum(z[k, :] @ Q @ z[k, :] for k in range(1,N+1))
    obj2 = sum(u[k, :] @ R @ u[k, :] for k in range(N))
    m.setObjective(obj1 + obj2, GRB.MINIMIZE)
    #m.params.NonConvex = 2
    if DEBUG == False:
        m.Params.LogToConsole = 0
        m.Params.OutputFlag = 0
    m.optimize()

    opt_status = m.status
    if env is None:
        print('Optimization status:',opt_status)
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
        #control = np.array([[umax[0][0]/100]])
        objval = 0
    else:
        control = u.X[0:1,:]/100
        objval = m.objval
    if x0[4] >= pusher.block_length/2 or x0[4] <= -pusher.block_length/2:
        print("Problem with the block height")
    #control = np.array([[control[0][1],control[0][0]]])
    #control = np.array([[0.015,0]])
    return control, objval, opt_status

if __name__ == '__main__':
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    dir = 'results/push_'+str(T)+'_'
    SAVE = True
    kmpc_run = True
    num_states = 6
    dt = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    max_timesteps = 200

    conduit_traj = np.zeros((max_timesteps,num_states+2))
    conduit_traj[:max_timesteps,0] = np.arange(0,max_timesteps)*3*0.7*dt/max_timesteps

    # Load the model
    num_obs = 100
    model = NeuralNetwork(N_x = num_states, N_e = num_obs, N_h = 32)
    checkpoint_stab = torch.load('nn_alt_loss/model.pt')
    model.load_state_dict(checkpoint_stab['model_dict'])
    model.to(device)

    # Set initial values
    block_length = 0.064
    mass_block = 0.029
    mu_t = 0.35
    #table_mu = 0.9
    mu_p = 0.3
    max_energy = None
    init_state = np.array(
        [0, 0.05, 0, -(block_length/2), 0, 0, 0, 0])
    pusher = Pusher(block_length, mass_block, mu_t, mu_p, init_state)
    

    # Extract the A matrix from the model
    A = model.A.weight.detach().cpu().numpy()
    print('A shape:', A.shape)
    B_diag = np.diag(np.array([1,1]))
    B = np.zeros((2,A.shape[1]))
    B[:,1:3] = B_diag
    B = B.T
    print('B shape:', B.shape)
    print(B)
    
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
    Q_overall[2,2] = 1e2
    #Q_overall[2,2] = 1e10
    Q_overall[0,0] = 1e5
    R_overall = np.eye(2)*dt*1e-5

    # Steady Push QR
    Q = np.eye(num_states+2)*0 # 0.003 cost total
    Q[0,0] = 1e2
    Q[1,1] = 1e2
    Q[2,2] = 1e0
    #Q[2,2] = 1e-5
    #Q[4,4] = 1e3
    # Q[3,3] = 1e1
    # Q[4,4] = 1e1
    #Q[5,5] = 1e1
    #R = np.eye(2)*dt*9e-4
    R = np.eye(2)*dt*7e-4
    R[1,1] = 9e-3

    Q_overall = Q
    R_overall = R

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
    xi_pred_arr = []
    xi_pred_arr.append(np.tile(xi,(T+1,1)))

    kmpc_times = []
    full_times = []
    print('Starting controller...') 
    range_val = conduit_traj.shape[0]
    #range_val = max_timesteps
    CONTROLLED_LIST = []
    UNCONTROLLED_LIST = []
    xi_nospeed = xi.copy()
    for step in range(max_timesteps-1):
        if step != 0:
            full_start_time = time.time()
        conduit_traj_ref = conduit_traj[step,:]
        if step <  range_val-T:
            conduit_traj_ref_arr = conduit_traj[step:step+T+1,:]
            # Convert the velocities to be in the body frame
            xi_ref = calculate_xi_wpos(conduit_traj_ref_arr, model)

        else:
            conduit_traj_ref_arr = conduit_traj_ref*np.ones((T+1,num_states+2))
            # Convert the velocities to be in the body frame
            xi_ref = calculate_xi_wpos(conduit_traj_ref_arr, model)

        if kmpc_run:
            theta = xi[2]
            #xi = xi_nospeed
            if step != 0:
                kmpc_start_time = time.time()
                control, xi_pred, objval, opt_status, m = mpc_gurobi(xi, xi_ref, m, u,x_var, x0_constr, z_constr_arr)
                kmpc_end_time = time.time()
                kmpc_times.append(kmpc_end_time-kmpc_start_time)
            else:
                control, xi_pred, objval, opt_status, m, u, x_var, x0_constr, z_constr_arr = mpc_gurobi_create_model(xi, xi_ref,A=A,B=B, DEBUG=False,Q_base=Q,R_base=R)

            control = control[0]
            control_old = control
            body_to_xy = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T
            control = control @ body_to_xy # The control is in the local coordinates
            if opt_status == 12:
                NUMERICAL_ERROR_LIST.append(step)
            if step != 0:
                full_end_time = time.time()
                full_times.append(full_end_time-full_start_time)
 
        
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
        cost = z[0]**2
        cost_arr.append(cost)
        res = [xi[:num_states+2], control]
        res_arr.append(res)
        if MPC_cost > 1000:
            print("High MPC cost")
        xi_nospeed = xi.copy()
        xi_nospeed[5:8] = 0.
        xi_pred_arr.append(xi_pred)

    print('Average KMPC time:', np.mean(kmpc_times))
    print('Total Cost:', np.sum(cost_arr))
    print('MPC Cost:',np.sum(MPC_cost_arr))
    print('How many numerical errors?', len(NUMERICAL_ERROR_LIST))

    # Plot the control inputs
    control_data = np.array([res[1] for res in res_arr])
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
    print("state data shape:", state_data.shape)
    pusher.animate_trajectory_comparison(pusher.Convert_to_xy(list(state_data)), conduit_traj, dt, SAVE=SAVE,str=dir,lims=[-0.1,0.55,-0.1,0.55])
    state_pred_arr = []
    for pred in xi_pred_arr:
        if pred is not None:
            state_pred_arr.append(pred[:,:num_states+2])
        else:
            state_pred_arr.append(np.zeros((xi_pred_arr[0].shape[0],num_states+2)))

    fig, axs = plt.subplots(4,3, figsize=(15, 10))
    fig.suptitle('System States')
    for i in range(num_states+2):
        axs[i//3, i%3].plot(state_data[:,i])
        axs[i//3, i%3].set_title('State ' + str(i))
        # Plot x_ref as well
        axs[i//3, i%3].plot(conduit_traj[:,i], color='r', linestyle='--')
        # Plot when the state was uncontrolled
        for uncontrolled_step in UNCONTROLLED_LIST:
            axs[i//3, i%3].axvline(x=uncontrolled_step, color='g', linestyle='dotted')
    if SAVE:
        plt.savefig(dir+'state_plot.png')
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
    print("Done.")

    # Plot the kmpc times on a seaborn violinplot
    np.save(dir+'kmpc_times.npy', kmpc_times)
    np.save(dir+'full_times.npy', full_times)
    plt.figure()
    sns.violinplot([full_times])
    plt.title('Square Pusher Runtimes')
    plt.ylabel('Time (s)')
    if SAVE:
        plt.savefig(dir+'runtime_violinplot.png')
    plt.show()
    print("Done.")