import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path

import time

import numpy as np
import casadi as ca

from mpc_controller import MPC_HOCBF
from config import *
from utils import plot_traj, save_animation_unicycle_trajectory, save_unicycle_final_shot

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))



def simulation(x_0, x_goal, obstacles):
    x_k = np.asarray(x_0, dtype=float).reshape(4,)
    x_goal = np.asarray(x_goal, dtype=float).reshape(4,)

    state_traj = [x_k.copy()]
    control_traj = []

    for k in range(total_steps_sim):
        u_k = controller.get_control_input(ca.DM(x_k), ca.DM(x_goal), obstacles)

        control_traj.append(np.array(u_k, dtype=float).reshape(2,))
        x_k = controller.rk4(ca.DM(x_k), ca.DM(u_k)).full().reshape(4,)
        x_k[2] = wrap_to_pi(x_k[2])
        state_traj.append(x_k.copy())

    
        print(f"mpc_hocbf N={N} - timestep {k} finished")

    return np.array(state_traj), np.array(control_traj)


if __name__ == "__main__":
    controller = MPC_HOCBF(dt, N, a_min, a_max, w_min, w_max, alpha0, alpha1)

    # 4-state initial conditions
    state_0a = np.array([-1.9, 2.0, -1.79, 0.0], dtype=float)  
    state_0b = np.array([-5.24, 4.11, 2.72, 0.0], dtype=float)
    state_0c = state_0b

    # choose scenarios
    initial_state_option = 'c'   # 'a', 'b', or 'c'

    # CASE SELECTION
    if initial_state_option == 'a':
        x_0 = state_0a
        x_goal = np.array([0.0, 0.0, 0.0, 0.0])      # zero reference state

        obstacles = []  # case a with no obstacle   (to test without obstacles uncomment out this line and comment the line below)
       # obstacles = [{"xo": -0.5, "yo": 1.2, "r": 0.1}, {"xo": -1.5, "yo": 1.2, "r": 0.1},]  # case a with 2 obstacles 

    elif initial_state_option == 'b':
        x_0 = state_0b
        x_goal = np.array([0.0, 0.0, 0.0, 0.0])

        obstacles = []  # case b with no obstacle      (to test without obstacles uncomment out this line and comment the line below)
    
        #obstacles = [{"xo": -4, "yo":2, "r": 0.5}, {"xo": -1.5, "yo":1.5, "r": 0.5}, {"xo": -2.5, "yo":3.0, "r": 0.5}]   # for case b with 3 obstacles
   

    else:  # case c
        x_0 = state_0c
        x_goal = np.array([1.0, 1.0, 0.0, 0.0])  # non-zero reference state

        obstacles = []  # case c with no obstacle       (to test without obstacles uncomment out this line and comment the line below)
        
       # obstacles = [{"xo": -3.5, "yo":2, "r": 0.5}, {"xo": -1.0, "yo":1.5, "r": 0.5}, {"xo": -2.0, "yo":3.0, "r": 0.5}]   # for case c with 3 obstacles

        # Define non-zero reference state
        x_ref = 1; y_ref = 1; theta_ref = 0; v_ref = 0

if obstacles:
    obs_tag = f"obs{len(obstacles)}"
else:
    obs_tag = "no_obs"


# For plotting/animation: hide obstacles when none are present.
obstacles_for_plot = obstacles if len(obstacles) > 0 else None

start_time = time.time()
state_traj, control_traj = simulation(x_0, x_goal, obstacles)
end_time = time.time()

execution_time = end_time - start_time
time_per_step = execution_time / total_steps_sim
print(f"Simulation executed in {execution_time:.2f}s, time per step: {time_per_step:.4f}s")

time_grid = np.arange(0, len(state_traj) * dt, dt)

plot_option = f"{initial_state_option}_{obs_tag}"
plot_traj(state_mpc=state_traj, u_mpc=control_traj, time=time_grid, h=50, option=plot_option)

x_mpc = state_traj[:, 0]
y_mpc = state_traj[:, 1]
theta_mpc = state_traj[:, 2]
v_mpc = state_traj[:, 3]
a_mpc = control_traj[:, 0]
w_mpc = control_traj[:, 1]

# Compute trajectory gradients (numerical derivatives)
x_traj_grad = np.gradient(x_mpc, dt)
y_traj_grad = np.gradient(y_mpc, dt)
theta_traj_grad = np.gradient(theta_mpc, dt)
v_traj_grad = np.gradient(v_mpc, dt)
a_traj_grad = np.gradient(a_mpc, dt)
w_traj_grad = np.gradient(w_mpc, dt)

# Compute mean squared derivative
x_traj_msd = np.mean(x_traj_grad ** 2)
y_traj_msd = np.mean(y_traj_grad ** 2)
theta_traj_msd = np.mean(theta_traj_grad ** 2)
v_traj_msd = np.mean(v_traj_grad ** 2)
avg_state_msd = (x_traj_msd + y_traj_msd + theta_traj_msd + v_traj_msd) / 4
print(f"Average State Trajectory Mean Squared derivatives {avg_state_msd:.2f}")

a_traj_msd = np.mean(a_traj_grad ** 2)
w_traj_msd = np.mean(w_traj_grad ** 2)
avg_u_msd = (a_traj_msd + w_traj_msd) / 2
print(f"Average Control Input Trajectory Mean Squared derivatives {avg_u_msd:.2f}")

# Calculate absolute convergence error
theta_err = wrap_to_pi(theta_mpc[-1] - x_goal[2])
if initial_state_option == 'c':
    abs_convergence_err = ( abs(x_mpc[-1] - x_ref) + abs(y_mpc[-1] - y_ref) + abs(theta_err) + abs(v_mpc[-1] - v_ref))
else:
    abs_convergence_err = abs(x_mpc[-1]) + abs(y_mpc[-1]) + abs(theta_err) + abs(v_mpc[-1])

print(f'Final state: [{x_mpc[-1]:.2f}; {y_mpc[-1]:.2f}; {theta_mpc[-1]:.2f} ; {v_mpc[-1]:.2f}]')
print(f'Absolute convergence error: {abs_convergence_err:.2f}')

file_name = f'mpc_N{N}_{initial_state_option}_{obs_tag}'

save_animation_unicycle_trajectory(x_mpc,y_mpc,theta_mpc,initial_state_option,file_name, obstacles=obstacles_for_plot, robot_r=0.25, margin=0.05,)

save_unicycle_final_shot(x_mpc,y_mpc, file_name=f"final_shot_{file_name}", start_xy=(float(x_0[0]), float(x_0[1])), goal_xy=(float(x_goal[0]), float(x_goal[1])), theta_robot=theta_mpc,
    obstacles=obstacles_for_plot,  robot_r=0.25, margin=0.05, show_safety_boundary=True, draw_robot=True,  robot_alpha=0.45,)




