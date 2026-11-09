import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import time

import numpy as np

from config import *
from mpc_controller_acados_rti import MPC_ACADOS_RTI
from utils import plot_traj, save_animation_unicycle_trajectory, save_unicycle_final_shot


BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

ROBOT_R = 0.25
SAFETY_MARGIN = 0.05


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def obstacle_r_eff(obs):
    return float(obs["r"]) + ROBOT_R + SAFETY_MARGIN


def hocbf_diagnostics_value(state, control, obs):
    x, y, theta, v = np.asarray(state, dtype=float).reshape(4,)
    a, w = np.asarray(control, dtype=float).reshape(2,)
    xo = float(obs["xo"])
    yo = float(obs["yo"])
    r_eff = obstacle_r_eff(obs)

    px = x - xo
    py = y - yo
    h_value = px * px + py * py - r_eff * r_eff

    s = px * np.cos(theta) + py * np.sin(theta)
    t = -px * np.sin(theta) + py * np.cos(theta)

    dh = 2 * v * s
    ddh = 2 * s * a + 2 * v * t * w + 2 * v * v
    hocbf_value = ddh + alpha1 * dh + alpha0 * h_value

    return h_value, hocbf_value


def print_safety_diagnostics(state_traj, control_traj, obstacles, tol=1e-6):
    if not obstacles:
        print("Safety diagnostics: no obstacles to check.")
        return

    state_traj = np.asarray(state_traj, dtype=float)
    control_traj = np.asarray(control_traj, dtype=float).reshape(-1, 2)

    min_h = np.inf
    min_h_info = None
    for k, state in enumerate(state_traj):
        for obs_idx, obs in enumerate(obstacles):
            x, y = state[0], state[1]
            dx = x - float(obs["xo"])
            dy = y - float(obs["yo"])
            r_eff = obstacle_r_eff(obs)
            dist = np.hypot(dx, dy)
            h_value = dist * dist - r_eff * r_eff
            if h_value < min_h:
                min_h = h_value
                min_h_info = (k, obs_idx, dist - r_eff)

    h_status = "VIOLATION" if min_h < -tol else "OK"
    h_step, h_obs_idx, h_clearance = min_h_info
    print(f"Safety diagnostics ({h_status}): min h={min_h:.6f} "
        f"at step {h_step}, obstacle {h_obs_idx}, clearance={h_clearance:.6f} m")

    if len(control_traj) == 0:
        print("HOCBF diagnostics: no applied controls to check.")
        return

    min_hocbf = np.inf
    min_hocbf_info = None
    for k, control in enumerate(control_traj):
        for obs_idx, obs in enumerate(obstacles):
            _, hocbf_value = hocbf_diagnostics_value(state_traj[k], control, obs)
            if hocbf_value < min_hocbf:
                min_hocbf = hocbf_value
                min_hocbf_info = (k, obs_idx)

    hocbf_status = "VIOLATION" if min_hocbf < -tol else "OK"
    hocbf_step, hocbf_obs_idx = min_hocbf_info
    print(
        f"HOCBF diagnostics ({hocbf_status}): min applied HOCBF={min_hocbf:.6f} "
        f"at step {hocbf_step}, obstacle {hocbf_obs_idx}"
    )


def save_failure_artifacts( state_traj, control_traj, x_0, x_goal, obstacles,  initial_state_option, file_name,):
    state_traj = np.asarray(state_traj, dtype=float)
    control_traj = np.asarray(control_traj, dtype=float).reshape(-1, 2)
    obstacles_for_plot = obstacles if len(obstacles) > 0 else None

    if len(control_traj) > 0:
        try:
            time_grid = np.arange(0, len(state_traj) * dt, dt)
            plot_traj( state_mpc=state_traj, u_mpc=control_traj, time=time_grid, h=50, option=file_name,)
        except Exception as exc:
            print(f"Could not save failure state/control plot: {exc}")
    else:
        print("Skipping failure state/control plot: no control input was computed before failure.")

    try:
        save_animation_unicycle_trajectory( state_traj[:, 0], state_traj[:, 1], state_traj[:, 2], initial_state_option, file_name, obstacles=obstacles_for_plot,
            robot_r=ROBOT_R,
            margin=SAFETY_MARGIN  )
        
    except Exception as exc:
        print(f"Could not save failure unicycle animation: {exc}")

    try:
        save_unicycle_final_shot(state_traj[:, 0], state_traj[:, 1], file_name=f"final_shot_{file_name}",
            start_xy=(float(x_0[0]), float(x_0[1])),
            goal_xy=(float(x_goal[0]), float(x_goal[1])),
            theta_robot=state_traj[:, 2],
            obstacles=obstacles_for_plot,
            robot_r=ROBOT_R,
            margin=SAFETY_MARGIN,
            show_safety_boundary=True,
            draw_robot=True,
            robot_alpha=0.45,
        )
    except Exception as exc:
        print(f"Could not save failure final-shot image: {exc}")


def simulation(x_0, x_goal, obstacles, initial_state_option, failure_artifacts_name=None,):
    x_k = np.asarray(x_0, dtype=float).reshape(4,)
    x_goal = np.asarray(x_goal, dtype=float).reshape(4,)

    state_traj = [x_k.copy()]
    control_traj = []

    controller.warm_start(x_k, x_goal)
    for k in range(total_steps_sim):
        try:
            u_k = controller.get_control_input(x_k, x_goal, obstacles)
        except RuntimeError:
            if failure_artifacts_name is not None:
                save_failure_artifacts(
                    state_traj,
                    control_traj,
                    x_0,
                    x_goal,
                    obstacles,
                    initial_state_option,
                    file_name=f"{failure_artifacts_name}_step{k}",
                )
                print(f"Saved failure artifacts at timestep {k}")
            raise

        u_vec = np.asarray(u_k, dtype=float).reshape(2,)

        control_traj.append(u_vec.copy())
        x_k = controller.rk4(x_k, u_vec)
        x_k[2] = wrap_to_pi(x_k[2])
        state_traj.append(x_k.copy())

        print(f"mpc_acados_rti_hocbf N={N} - timestep {k} finished")

    return np.asarray(state_traj, dtype=float), np.asarray(control_traj, dtype=float)


if __name__ == "__main__":
    # 4-state initial conditions
    state_0a = np.array([-1.9, 2.0, -1.79, 0.0], dtype=float)
    state_0b = np.array([-5.24, 4.11, 2.72, 0.0], dtype=float)
    state_0c = state_0b

    # choose scenarios
    initial_state_option = "a"   # "a", "b", or "c"


    # CASE SELECTION
    if initial_state_option == "a":
        x_0 = state_0a
        x_goal = np.array([0.0, 0.0, 0.0, 0.0])

        obstacles = []  # case a with no obstacle
        obstacles = [{"xo": -0.5, "yo": 1.2, "r": 0.1}, {"xo": -1.5, "yo": 1.2, "r": 0.1},]  # case a with 2 obstacles

    elif initial_state_option == "b":
        x_0 = state_0b
        x_goal = np.array([0.0, 0.0, 0.0, 0.0])

        obstacles = []  # case b with no obstacle
        obstacles = [{"xo": -4, "yo": 2, "r": 0.5}, {"xo": -1.5, "yo": 1.5, "r": 0.5}, {"xo": -2.5, "yo": 3.0, "r": 0.5}]  # case b with 3 obstacles
    
    else:  # case c
        x_0 = state_0c
        x_goal = np.array([1.0, 1.0, 0.0, 0.0])

        obstacles = []  # case c with no obstacle
        obstacles = [{"xo": -3.5, "yo": 2, "r": 0.5}, {"xo": -1.0, "yo": 1.5, "r": 0.5}, {"xo": -2.0, "yo": 3.0, "r": 0.5}]  # case c with 3 obstacles

        # Define non-zero reference state
        x_ref = 1; y_ref = 1; theta_ref = 0; v_ref = 0
    controller = MPC_ACADOS_RTI(dt, N, a_min, a_max, w_min, w_max,alpha0, alpha1,max_obstacles=len(obstacles),)

if obstacles:
    obs_tag = f"obs{len(obstacles)}"
else:
    obs_tag = "no_obs"


# For plotting/animation: hide obstacles when none are present.
obstacles_for_plot = obstacles if len(obstacles) > 0 else None

start_time = time.time()
failure_artifacts_name = f"failure_mpc_acados_rti_N{N}_{initial_state_option}_{obs_tag}"
state_traj, control_traj = simulation(x_0, x_goal, obstacles, initial_state_option, failure_artifacts_name=failure_artifacts_name,)
end_time = time.time()

execution_time = end_time - start_time
time_per_step = execution_time / total_steps_sim
print(f"Simulation executed in {execution_time:.2f}s, time per step: {time_per_step:.4f}s")

time_grid = np.arange(0, len(state_traj) * dt, dt)
print_safety_diagnostics(state_traj, control_traj, obstacles)

plot_option = f"acados_rti_{initial_state_option}_{obs_tag}"
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
if initial_state_option == "c":
    abs_convergence_err = abs(x_mpc[-1] - x_ref) + abs(y_mpc[-1] - y_ref) + abs(theta_err) + abs(v_mpc[-1] - v_ref)
else:
    abs_convergence_err = abs(x_mpc[-1]) + abs(y_mpc[-1]) + abs(theta_err) + abs(v_mpc[-1])

print(f"Final state: [{x_mpc[-1]:.2f}; {y_mpc[-1]:.2f}; {theta_mpc[-1]:.2f} ; {v_mpc[-1]:.2f}]")
print(f"Absolute convergence error: {abs_convergence_err:.2f}")

file_name = f"mpc_acados_rti_N{N}_{initial_state_option}_{obs_tag}"

save_animation_unicycle_trajectory(x_mpc,y_mpc, theta_mpc, initial_state_option, file_name, obstacles=obstacles_for_plot, robot_r=ROBOT_R, margin=SAFETY_MARGIN,)

save_unicycle_final_shot(x_mpc, y_mpc, file_name=f"final_shot_{file_name}", start_xy=(float(x_0[0]), float(x_0[1])), goal_xy=(float(x_goal[0]), float(x_goal[1])),
    theta_robot=theta_mpc, obstacles=obstacles_for_plot, robot_r=ROBOT_R, margin=SAFETY_MARGIN, show_safety_boundary=True, draw_robot=True, robot_alpha=0.45,)
