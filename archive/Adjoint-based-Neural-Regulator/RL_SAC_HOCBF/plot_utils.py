from HOCBF_SAC.utils import to_tensor, to_numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from tqdm import tqdm
from HOCBF_SAC.utils import prYellow
import os
import json

def plot_value_function(env, agent, dynamics_model, save_path='', safe_action=False):

    fn = save_path + '/figures/vf_plot.png'
    if save_path and os.path.exists(fn):
        prYellow('Not plotting value function since figure already exists at {}'.format(fn))
        return

    if env.dynamics_mode == 'Unicycle':
        res = 20
        xs = np.linspace(env.bds[0][0], env.bds[1][0], res)
        ys = np.linspace(env.bds[0][1], env.bds[1][1], res)
        ths = np.linspace(-np.pi, np.pi, res)
        xxs, yys, thhs = np.meshgrid(xs, ys, ths)
        # Use zero speed for value visualization slices with the 4D state [x, y, theta, v].
        states = np.vstack((xxs.ravel(), yys.ravel(), thhs.ravel(), np.zeros_like(thhs).ravel())).transpose()
        actions = np.zeros((states.shape[0], env.action_space.shape[0]))
        obs = np.zeros((states.shape[0], env.observation_space.shape[0]))
        # Get Actions and Observations corresponding to Each State (TODO:can be vectorized if env is vectorized...)
        for i in tqdm(range(states.shape[0])):
            env.state = states[i]
            obs[i] = env.get_obs()
            actions[i], _ = agent.select_action(obs[i], dynamics_model, evaluate=True, safe_action=safe_action)
        obs = to_tensor(obs, torch.FloatTensor, agent.device)
        actions = to_tensor(actions, torch.FloatTensor, agent.device)
        vf1, vf2 = agent.critic(obs, actions)  # Each is Nx3
        vf1 = to_numpy(vf1).squeeze()
        vf2 = to_numpy(vf2).squeeze()
        vf = np.min(np.vstack((vf1, vf2)).transpose(), axis=1, keepdims=True)
        # Take Max over thetas
        vf = np.max(vf.reshape((res, res, res)), axis=2, keepdims=False)
        fig, ax = plt.subplots()

        # Add obstacles, goal and initial pos
        for i in range(len(env.hazards)):
            if env.hazards[i]['type'] == 'circle':
                ax.add_patch(plt.Circle(env.hazards[i]['location'], env.hazards[i]['radius'], color='r', fill=False, linewidth=2.0))
            

        ax.add_patch(plt.Circle(env.goal_pos, env.goal_size, color='g', fill=False, linewidth=2.0))
        for initial_state in env.initial_state:
            ax.add_patch(plt.Circle(initial_state[:2], 0.3, color='b', fill=False, linewidth=2.0))

        c = ax.pcolormesh(xs, ys, vf.reshape((res, res)), cmap='hot', shading='nearest', vmin=1.0, vmax=2.0, alpha=0.8)
        fig.colorbar(c, ax=ax)
        if save_path:
            if not os.path.exists(save_path + '/figures'):
                os.mkdir(save_path + '/figures')
            plt.savefig(save_path + '/figures/vf_plot.png')
        else:
            plt.show()
        plt.close()


def save_animation_unicycle_trajectory( x_robot, y_robot, theta_robot,  initial_state_option,  gif_name, obstacles=None, robot_r=0.25,  margin=0.05,):
    os.makedirs("./animation/unicycle", exist_ok=True)
    if initial_state_option == "a":
        start_pos = [-1.9, 2.0, -1.79, 0.0]
    else:
        start_pos = [-5.24, 4.11, 2.72, 0.0]

    if initial_state_option == "c":
        end_pos = [1, 1, 0, 0]
    else:
        end_pos = [0, 0, 0, 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    x_candidates = [float(np.min(x_robot)), float(np.max(x_robot)), float(start_pos[0]), float(end_pos[0])]
    y_candidates = [float(np.min(y_robot)), float(np.max(y_robot)), float(start_pos[1]), float(end_pos[1])]
    if obstacles is not None:
        for obs in obstacles:
            xo = float(obs["xo"])
            yo = float(obs["yo"])
            r_eff = float(obs["r"]) + robot_r + margin
            x_candidates.extend([xo - r_eff, xo + r_eff])
            y_candidates.extend([yo - r_eff, yo + r_eff])

    viewport_pad = 0.75
    ax.set_xlim(min(x_candidates) - viewport_pad, max(x_candidates) + viewport_pad)
    ax.set_ylim(min(y_candidates) - viewport_pad, max(y_candidates) + viewport_pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.3)

    if obstacles is not None:
        for j, obs in enumerate(obstacles):
            xo = float(obs["xo"])
            yo = float(obs["yo"])
            r_obs = float(obs["r"])
            r_eff = r_obs + robot_r + margin
            ax.add_patch( plt.Circle(  (xo, yo), r_obs, fill=False,  linewidth=3, color="black",  label="Obstacle" if j == 0 else None,  zorder=4,  ) )
            ax.add_patch(
                plt.Circle( (xo, yo), r_eff, fill=False, linewidth=2, linestyle="--", color="red", label="Safety boundary" if j == 0 else None, zorder=3, ) )

    robot_body = plt.Circle((0, 0), robot_r, color="cyan", fill=True, linewidth=2)
    wheel_width = 0.1
    wheel_height = 0.2
    wheel1 = plt.Rectangle((-wheel_height / 2, -wheel_width / 2),  wheel_height,  wheel_width,  color="gold", )
    wheel2 = plt.Rectangle( (-wheel_height / 2, -wheel_width / 2), wheel_height, wheel_width, color="gold", )
    ax.add_patch(robot_body)
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)

    start_xy = plt.Circle((start_pos[0], start_pos[1]),0.15, color="green", fill=True)
    goal_xy = plt.Circle((end_pos[0], end_pos[1]), 0.15, color="red", fill=True)
    ax.add_patch(start_xy)
    ax.add_patch(goal_xy)

    robot_path, = ax.plot([], [], linestyle="--", color="saddlebrown", label="Traveled Path", linewidth=2 )
    ax.legend(loc="upper right", fontsize=20)

    def update(frame):
        x, y, theta = x_robot[frame], y_robot[frame], theta_robot[frame]
        robot_body.center = (x, y)
        wheel_offset = 0.3
        wheel1_center_x = x - wheel_offset * np.sin(theta)
        wheel1_center_y = y + wheel_offset * np.cos(theta)
        wheel1.set_xy(  (  wheel1_center_x - wheel_height / 2 * np.cos(theta) + wheel_width / 2 * np.sin(theta),  wheel1_center_y   - wheel_height / 2 * np.sin(theta)
                - wheel_width / 2 * np.cos(theta),  ) )
        wheel1.angle = np.degrees(theta)
        wheel2_center_x = x + wheel_offset * np.sin(theta)
        wheel2_center_y = y - wheel_offset * np.cos(theta)
        wheel2.set_xy( ( wheel2_center_x - wheel_height / 2 * np.cos(theta) + wheel_width / 2 * np.sin(theta), wheel2_center_y - wheel_height / 2 * np.sin(theta)
                - wheel_width / 2 * np.cos(theta),  ) )
        
        wheel2.angle = np.degrees(theta)
        robot_path.set_data(x_robot[: frame], y_robot[: frame])
        return robot_body, wheel1, wheel2, robot_path

    time_per_step = 0.05
    plt.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=len(x_robot), blit=True)
    output_dir = f"./animation/unicycle/{gif_name}.mp4"
    anim.save(output_dir, writer="ffmpeg", fps=1 / time_per_step)
    print(f"Animation (wheeled robot motion) saved to {output_dir}")
    plt.close(fig)


def save_unicycle_final_shot(x_robot, y_robot, file_name,start_xy=None, goal_xy=None, theta_robot=None, obstacles=None, robot_r=0.25,margin=0.05, 
                             show_safety_boundary=True, draw_robot=True, robot_alpha=0.45,):
    """
    Save a static snapshot of the executed trajectory with optional safety boundaries
    and a unicycle rendering at the terminal state.
    Optional start_xy/goal_xy force fixed markers and viewport inclusion.
    """
    os.makedirs("./figs/final_shot", exist_ok=True)

    x_robot = np.asarray(x_robot).reshape(-1)
    y_robot = np.asarray(y_robot).reshape(-1)
    if theta_robot is not None:
        theta_robot = np.asarray(theta_robot).reshape(-1)
        if len(theta_robot) != len(x_robot):
            raise ValueError("theta_robot must have the same length as x_robot/y_robot.")

    if len(x_robot) == 0 or len(y_robot) == 0:
        raise ValueError("x_robot and y_robot must be non-empty.")

    traj_start_xy = (float(x_robot[0]), float(y_robot[0]))
    end_xy = (float(x_robot[-1]), float(y_robot[-1]))
    end_theta = float(theta_robot[-1]) if theta_robot is not None else 0.0
    start_marker_xy = (float(start_xy[0]), float(start_xy[1]), ) if start_xy is not None else traj_start_xy
    goal_marker_xy = (float(goal_xy[0]), float(goal_xy[1]), ) if goal_xy is not None else end_xy

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x_robot, y_robot, color="tab:blue", linewidth=2.5, label="Trajectory", zorder=3)

    x_candidates = [ float(np.min(x_robot)), float(np.max(x_robot)),  start_marker_xy[0],  goal_marker_xy[0],   ]
    y_candidates = [ float(np.min(y_robot)), float(np.max(y_robot)),  start_marker_xy[1],  goal_marker_xy[1],   ]

    if obstacles is not None:
        for j, obs in enumerate(obstacles):
            xo = float(obs["xo"])
            yo = float(obs["yo"])
            r_obs = float(obs["r"])
            r_eff = r_obs + robot_r + margin

            ax.scatter([xo],[yo],color="gray",s=45,alpha=0.5,label="Obstacle" if j == 0 else "_nolegend_",zorder=1,)
            ax.add_patch(plt.Circle((xo, yo),r_obs,color="gray",alpha=0.8,label="_nolegend_" if j == 0 else None, zorder=1,))
            if show_safety_boundary:
                ax.add_patch( plt.Circle( (xo, yo), r_eff, fill=False, linestyle="--",  linewidth=1.5,  color="red", alpha=0.9,
                        label="_nolegend_" if j == 0 else None,  zorder=2,  )    )

            visible_r = r_eff if show_safety_boundary else r_obs
            x_candidates.extend([xo - visible_r, xo + visible_r])
            y_candidates.extend([yo - visible_r, yo + visible_r])

    ax.scatter([start_marker_xy[0]], [start_marker_xy[1]], color="green", s=45, alpha=0.5, label="Start", zorder=4)
    ax.scatter([goal_marker_xy[0]], [goal_marker_xy[1]], color="red", s=45, alpha=0.5, label="Goal", zorder=4)
    ax.add_patch(plt.Circle( (start_marker_xy[0], start_marker_xy[1]), 0.10, color="green", fill=True, alpha=0.5, label="_nolegend_",  zorder=4,  )  )
    ax.add_patch( plt.Circle((goal_marker_xy[0], goal_marker_xy[1]), 0.10, color="red",  fill=True, alpha=0.5, label="_nolegend_",  zorder=4,)  )

    if draw_robot:
        # Draw a simple unicycle footprint at the terminal state.
        body = plt.Circle(end_xy, robot_r, color="cyan", ec="black", lw=1.2, alpha=robot_alpha, zorder=5)
        ax.add_patch(body)

        wheel_length = 0.20
        wheel_width = 0.08
        wheel_offset = robot_r + 0.05

        c = np.cos(end_theta)
        s = np.sin(end_theta)
        fwd = np.array([c, s], dtype=float)
        lat = np.array([-s, c], dtype=float)

        def _wheel_polygon(center_xy):
            center = np.array(center_xy, dtype=float)
            p1 = center - 0.5 * wheel_length * fwd - 0.5 * wheel_width * lat
            p2 = center + 0.5 * wheel_length * fwd - 0.5 * wheel_width * lat
            p3 = center + 0.5 * wheel_length * fwd + 0.5 * wheel_width * lat
            p4 = center - 0.5 * wheel_length * fwd + 0.5 * wheel_width * lat
            return np.vstack([p1, p2, p3, p4])

        wheel_left_center = np.array(end_xy, dtype=float) + wheel_offset * lat
        wheel_right_center = np.array(end_xy, dtype=float) - wheel_offset * lat

        wheel_left = plt.Polygon( _wheel_polygon(wheel_left_center), closed=True, color="gold", ec="black", lw=0.8, alpha=robot_alpha, zorder=6, )
        wheel_right = plt.Polygon(_wheel_polygon(wheel_right_center), closed=True, color="gold", ec="black",lw=0.8, alpha=robot_alpha, zorder=6, ) 
        ax.add_patch(wheel_left)
        ax.add_patch(wheel_right)

    pad = 0.35
    ax.set_xlim(min(x_candidates) - pad, max(x_candidates) + pad)
    ax.set_ylim(min(y_candidates) - pad, max(y_candidates) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=24)
    plt.setp(ax.get_xticklabels(),fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best", prop={"weight": "bold", "size": 20})
    plt.tight_layout()

    output_path = f"./figs/final_shot/{file_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Final-shot figure saved to {output_path}")


def save_unicycle_trajectory_diagnostics(state_traj, action_traj, dt, ref_state, file_stem, output_root="./figs/diagnostics", plot_horizon_s=None, pad_plot_to_horizon=True,):
    os.makedirs(output_root, exist_ok=True)

    X = np.asarray(state_traj, dtype=float)
    U = np.asarray(action_traj, dtype=float)
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError("state_traj must have shape (T+1, 4).")
    if U.ndim != 2 or U.shape[1] != 2:
        raise ValueError("action_traj must have shape (T, 2).")
    if X.shape[0] != U.shape[0] + 1:
        raise ValueError("Expected len(state_traj) == len(action_traj) + 1.")

    # Keep metrics on the true rollout, but optionally pad plotting traces to a fixed horizon.
    X_plot = np.copy(X)
    U_plot = np.copy(U)
    if plot_horizon_s is not None:
        horizon_steps_ctrl = max(int(round(float(plot_horizon_s) / float(dt))), 0)
        horizon_steps_state = horizon_steps_ctrl + 1
        if pad_plot_to_horizon:
            if X_plot.shape[0] < horizon_steps_state:
                last_state = X_plot[-1:, :]
                pad_states = np.repeat(last_state, horizon_steps_state - X_plot.shape[0], axis=0)
                X_plot = np.vstack([X_plot, pad_states])
            if U_plot.shape[0] < horizon_steps_ctrl:
                if U_plot.shape[0] == 0:
                    last_u = np.zeros((1, 2), dtype=float)
                else:
                    last_u = U_plot[-1:, :]
                pad_inputs = np.repeat(last_u, horizon_steps_ctrl - U_plot.shape[0], axis=0)
                U_plot = np.vstack([U_plot, pad_inputs])

    x_traj = X[:, 0]
    y_traj = X[:, 1]
    theta_traj = X[:, 2]
    v_traj = X[:, 3]
    a_traj = U[:, 0]
    w_traj = U[:, 1]

    x_plot = X_plot[:, 0]
    y_plot = X_plot[:, 1]
    theta_plot = X_plot[:, 2]
    v_plot = X_plot[:, 3]
    a_plot = U_plot[:, 0]
    w_plot = U_plot[:, 1]

    t_state = np.arange(X_plot.shape[0]) * dt
    t_ctrl = np.arange(U_plot.shape[0]) * dt

    x_traj_grad = np.gradient(x_traj, dt)
    y_traj_grad = np.gradient(y_traj, dt)
    theta_traj_grad = np.gradient(theta_traj, dt)
    v_traj_grad = np.gradient(v_traj, dt)
    a_traj_grad = np.gradient(a_traj, dt)
    w_traj_grad = np.gradient(w_traj, dt)

    x_traj_msd = float(np.mean(x_traj_grad ** 2))
    y_traj_msd = float(np.mean(y_traj_grad ** 2))
    theta_traj_msd = float(np.mean(theta_traj_grad ** 2))
    v_traj_msd = float(np.mean(v_traj_grad ** 2))
    avg_state_msd = float((x_traj_msd + y_traj_msd + theta_traj_msd + v_traj_msd) / 4.0)

    a_traj_msd = float(np.mean(a_traj_grad ** 2))
    w_traj_msd = float(np.mean(w_traj_grad ** 2))
    avg_u_msd = float((a_traj_msd + w_traj_msd) / 2.0)

    x_norm_t = np.linalg.norm(X, axis=1)
    u_norm_t = np.linalg.norm(U, axis=1)
    x_traj_norm = float(np.linalg.norm(X))
    u_traj_norm = float(np.linalg.norm(U))
    Jx = float(np.sum(X ** 2))
    Ju = float(np.sum(U ** 2))

    final_state = X[-1]
    ref_state = np.asarray(ref_state, dtype=float).reshape(4,)
    abs_convergence_err = float(np.sum(np.abs(final_state - ref_state)))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(t_state, x_plot,  linestyle='-', dashes=[3, 1], label=r"$x_{RL}$", linewidth=5)
    plt.plot(t_state, y_plot,linestyle='-', dashes=[3, 1], label=r"$y_{RL}$", linewidth=5)
    plt.plot(t_state, theta_plot, linestyle='-', dashes=[3, 1], label=r"$\theta_{RL}$", linewidth=5)
    plt.plot(t_state, v_plot, linestyle='-', dashes=[3, 1], label=r"$v_{RL}$", linewidth=5)
    if plot_horizon_s is not None:
        plt.xlim(0.0, float(plot_horizon_s))
    plt.xlabel("Time (s)", fontsize=20, fontweight='bold')
    plt.ylabel("State Trajectory", fontsize=20, fontweight='bold')
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.xticks(fontsize=24,  fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')


    plt.subplot(1, 2, 2)
    plt.plot(t_ctrl, a_plot, linestyle='-', dashes=[3, 1], label=r"$a_{RL}$", linewidth=5)
    plt.plot(t_ctrl, w_plot, linestyle='-', dashes=[3, 1], label=r"$\omega_{RL}$", linewidth=5)
    if plot_horizon_s is not None:
        plt.xlim(0.0, float(plot_horizon_s))
    plt.xlabel("Time (s)", fontsize=20, fontweight='bold')
    plt.ylabel("Control Input", fontsize=20, fontweight='bold')
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.xticks(fontsize=24,  fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(output_root, f"{file_stem}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    metrics = {
        "final_state": final_state.tolist(),
        "ref_state": ref_state.tolist(),
        "absolute_convergence_error": abs_convergence_err,
        "avg_state_msd": avg_state_msd,
        "avg_control_msd": avg_u_msd,
        "state_2norm": x_traj_norm,
        "control_2norm": u_traj_norm,
        "state_energy_sum_sq": Jx,
        "control_energy_sum_sq": Ju,
        "state_norm_t": x_norm_t.tolist(),
        "control_norm_t": u_norm_t.tolist(),
    }

    metrics_path = os.path.join(output_root, f"{file_stem}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {"figure_path": fig_path, "metrics_path": metrics_path, "metrics": metrics,}
