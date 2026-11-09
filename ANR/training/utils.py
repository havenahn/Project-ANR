import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from networkx import omega
import numpy as np
from config import dt, N, T_sim, total_steps_sim
import casadi as ca
import torch
import random

def wrap_to_pi_torch(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))

class UnicycleDynamics(torch.nn.Module):
    def __init__(self):
        super(UnicycleDynamics, self).__init__()

    def forward(self, t, x_and_u):
        """
        Args:
            t: time (required by torchdiffeq, even if not used)
            x_and_u: concatenated state [x, y, theta,v] and control [a, omega] (torch tensor)

        Returns:
            dx/dt (torch tensor)
        """
        # Split state and control from concatenated tensor
        x = x_and_u[:, :4]   # [B,4]
        u = x_and_u[:, 4:]   # [B,2]

        theta = x[:, 2]
        v = x[:, 3]
        a = u[:, 0]
        omega = u[:, 1]

        x_dot = v * torch.cos(theta)
        y_dot = v * torch.sin(theta)
        theta_dot = omega
        v_dot = a

        a_dot = torch.zeros_like(a)
        omega_dot = torch.zeros_like(omega)

        dxdt = torch.stack([x_dot, y_dot, theta_dot, v_dot, a_dot, omega_dot], dim=1)                      # [B,6]
        return dxdt
    
def plot_traj(state_mpc, u_mpc, time, h,option):
    import os
    os.makedirs("./figs", exist_ok=True)
    x_mpc = state_mpc[:, 0]
    y_mpc = state_mpc[:, 1]
    theta_mpc = state_mpc[:,2]
    v_mpc    = state_mpc[:,3]

    a_mpc = u_mpc[:,0]
    w_mpc = u_mpc[:,1]

    # Plot State Trajectory
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    len_state_mpc = len(x_mpc)
    len_u_mpc = len(a_mpc)
    plt.plot(time[:len_state_mpc], x_mpc, linestyle='-', dashes=[3, 1], linewidth=5, label=r"$x_{mpc}$")
    plt.plot(time[:len_state_mpc], y_mpc, linestyle='-', dashes=[3, 1], linewidth=5, label=r"$y_{mpc}$")
    plt.plot(time[:len_state_mpc], theta_mpc, linestyle='-', dashes=[3, 1], linewidth=5, label=r"$\theta_{mpc}$")
    plt.plot(time[:len_state_mpc], v_mpc, linestyle='-', dashes=[3, 1], linewidth=5, label=r"$v_{mpc}$")
    plt.xlabel("Time (s)",fontsize=20, fontweight='bold')
    plt.ylabel("State Trajectory",fontsize=20, fontweight='bold')
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.xticks(fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')

    # Plot Control input trajectory
    plt.subplot(1, 2, 2)

    plt.plot(time[:len_u_mpc], a_mpc, linestyle='-', dashes=[3, 1], linewidth=5, label=r"$a_{mpc}$")
    plt.plot(time[:len_u_mpc], w_mpc, linestyle='-', dashes=[3, 1], linewidth=5, label=r"$w_{mpc}$")
    plt.xlabel("Time (s)",fontsize=20, fontweight='bold')
    plt.ylabel("Control Input",fontsize=20, fontweight='bold')
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.xticks(fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')
    plt.tight_layout()
    output_dir = f'./figs/mpc_N{N}_h{h}_{option}.png'
    plt.savefig(output_dir, dpi=300)
    plt.close()

    print(f"Figure saved to {output_dir}")
        
def dynamics(x, u):
    """Unicycle model dynamics
      x: shape (1,4) [x, y, theta, v]
    u: [a, omega] """

    theta = x[0,2]
    v     = x[0,3]
    a, omega = u[0], u[1]

    x_dot     = v * np.cos(theta)
    y_dot     = v * np.sin(theta)
    theta_dot = omega
    v_dot     = a
    return np.array([x_dot, y_dot, theta_dot, v_dot])

def rk4(x, u):
    # RK4 integration step for dynamics
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next[0,2] = wrap_to_pi_torch(torch.tensor(x_next[0,2])).item()
    return x_next

def solve_qp(lambda1, lambda2, lambda3,lambda4, v,theta,a_min, a_max, w_min, w_max):

    # Define decision variables
    a = ca.SX.sym('a')
    omega = ca.SX.sym('omega')
    

    # Define the Hamiltonian
    H = (a**2 + omega**2 +
        lambda1 * v* ca.cos(theta) +
        lambda2 * v * ca.sin(theta) +
        lambda3 * omega +
        lambda4 * a)

    # Set up the QP problem
    qp = {
        'x': ca.vertcat(a, omega),  # Decision variables [a, omega]
        'f': H,                    # Cost function (Hamiltonian)
        'g': ca.vertcat()          # No additional equality/inequality constraints
    }

    # Set bounds for the decision variables

    lbx = [a_min, w_min]  # Lower bounds for [a, omega]
    ubx = [a_max, w_max]    # Upper bounds for [a, omega]

    opts = {'printLevel': 'none'}  # Suppress solver output for qpoases 
    # Create QP solver
    S = ca.qpsol('S', 'qpoases', qp, opts)

    # Solve the problem
    solution = S(lbx=lbx, ubx=ubx)

    # Extract results
    a_opt = solution['x'][0]
    omega_opt = solution['x'][1]
    return a_opt, omega_opt


def hocbf_circle_coeffs(state_k, xo, yo, r_eff, k0, k1):
    """
    Returns coefficients so that:
        A @ [a, omega] >= b
    where A = [c_a, c_w] and b = -(c0 + k1*dh + k0*h)

    state_k is shape (1,4): [x, y, theta, v]
    """
    x = float(state_k[0,0])
    y = float(state_k[0,1])
    th = float(state_k[0,2])
    v  = float(state_k[0,3])

    px = x - xo
    py = y - yo

    h  = px*px + py*py - r_eff*r_eff

    s  = px*np.cos(th) + py*np.sin(th)
    dh = 2.0 * v * s
    c0  = 2.0 * v*v
    c_a = 2.0 * s
    c_w = 2.0 * v * (-px*np.sin(th) + py*np.cos(th))

    b = -(c0 + k1*dh + k0*h)   # so: c_a*a + c_w*omega >= b
    A = np.array([c_a, c_w], dtype=float)

    info = {"h": h, "dh": dh, "c0": c0, "c_a": c_a, "c_w": c_w, "b": b}
    return A, float(b), info


    
def solve_qp_hocbf(lambda1, lambda2, lambda3, lambda4,v,theta, state_k, obstacles, a_min, a_max, w_min, w_max, r1 =1, r2=1, 
                    k0=3.0, k1=3.0,robot_r=0.25, margin=0.05, rho=50, use_slack=False):
    """
    obstacles: list of dicts like [{"xo":.., "yo":.., "r":..}, ...]
    If use_slack=True, adds delta>=0 and relaxes all CBF constraints by +delta.
    """

    # Decision variables
    a = ca.SX.sym('a')
    omega = ca.SX.sym('omega')

    if use_slack:
        delta = ca.SX.sym('delta')
        x_dec = ca.vertcat(a, omega, delta)
    else:
        x_dec = ca.vertcat(a, omega)

    # Hamiltonian
    H = (a**2 + omega**2 + lambda1 * v* ca.cos(theta) + lambda2 * v * ca.sin(theta) + lambda3 * omega + lambda4 * a)
    if use_slack:
        H = H + rho*(delta**2)
    # Constraints g >= 0
    g_list = []
    infos = []

    for obs in obstacles:
        xo = float(obs["xo"])
        yo = float(obs["yo"])
        r_obs = float(obs["r"])
        r_eff = r_obs + robot_r + margin  # safety radius

        A, b, info = hocbf_circle_coeffs(state_k, xo, yo, r_eff, k0, k1)
        infos.append((obs, info))

        # A @ [a, omega] >= b
        g_i = float(A[0])*a + float(A[1])*omega - b  # >= 0 means satisfied
        if use_slack:
            g_i = g_i + delta
        g_list.append(g_i)

    if use_slack:
        # delta >= 0
        g_list.append(delta)

    g = ca.vertcat(*g_list) if len(g_list) > 0 else ca.vertcat()

    qp = {'x': x_dec, 'f': H, 'g': g}

    # Bounds on decision variables
    if use_slack:
        lbx = [float(a_min), float(w_min), 0.0]
        ubx = [float(a_max), float(w_max), ca.inf]
    else:
        lbx = [float(a_min), float(w_min)]
        ubx = [float(a_max), float(w_max)]

    # Bounds on constraints g >= 0
    if g.numel() > 0:
        lbg = [0.0]*int(g.numel())
        ubg = [ca.inf]*int(g.numel())
    else:
        lbg, ubg = [], []

    opts = {'printLevel': 'none'}
    S = ca.qpsol('S', 'qpoases', qp, opts)

    sol = S(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    a_opt = float(sol['x'][0])
    w_opt = float(sol['x'][1])
    if use_slack:
        d_opt = float(sol['x'][2])
    else:
        d_opt = 0.0

    return a_opt, w_opt, d_opt, infos




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_animation(t_span, x_traj, y_traj, theta_traj, v_traj, costate_trajectory, option):

    # Set up figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Left plot: State trajectory
    axs[0].set_xlabel("Time (s)",fontsize=20, fontweight='bold')
    axs[0].set_ylabel("State Trajectory",fontsize=20, fontweight='bold')
    line_state1, = axs[0].plot([], [], linestyle='-', dashes=[3, 1], label=r"$x_{anr}$", linewidth=5)
    line_state2, = axs[0].plot([], [], linestyle='-', dashes=[3, 1], label=r"$y_{anr}$", linewidth=5)
    line_state3, = axs[0].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\theta_{anr}$", linewidth=5)
    line_state4, = axs[0].plot([], [], linestyle='-', dashes=[3, 1], label=r"$v_{anr}$", linewidth=5)
    axs[0].legend(fontsize=28)
    axs[0].grid(True)
    axs[0].set_xlim(0, T_sim)
    axs[0].set_ylim(-5, 5)
    axs[0].tick_params(axis="both", labelsize=20)

    # Right plot: Co-state trajectory
    axs[1].set_xlabel("Time (s)", fontsize=20, fontweight='bold')
    axs[1].set_ylabel("Lambda values", fontsize=20, fontweight='bold')
    # Initialize co-state lines
    line_lambda1, = axs[1].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\lambda{x}$", linewidth=5)
    line_lambda2, = axs[1].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\lambda{y}$", linewidth=5)
    line_lambda3, = axs[1].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\lambda{\theta}$", linewidth=5)
    line_lambda4, = axs[1].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\lambda{v}$", linewidth=5)
    axs[1].legend(fontsize=28)
    axs[1].grid(True)
    axs[1].set_ylim(-10, 10)
    axs[1].tick_params(axis="both", labelsize=20)

    
    # Animation update function
    def update(frame):
        # Update state trajectory plot
        line_state1.set_data(t_span[:frame+1], x_traj[:frame+1])
        line_state2.set_data(t_span[:frame+1], y_traj[:frame+1])
        line_state3.set_data(t_span[:frame+1], theta_traj[:frame+1])
        line_state4.set_data(t_span[:frame+1], v_traj[:frame+1])
        # # Update co-state trajectory plot
        n = len( costate_trajectory[frame][0, :, 0])
        t_span_costate = np.linspace(frame*dt, (frame+n-1)*dt, n)
        axs[1].set_xlim(frame*dt, (frame+n-1)*dt)
        line_lambda1.set_data(t_span_costate, costate_trajectory[frame][0, :, 0])
        line_lambda2.set_data(t_span_costate, costate_trajectory[frame][0, :, 1])
        line_lambda3.set_data(t_span_costate, costate_trajectory[frame][0, :, 2])
        line_lambda4.set_data(t_span_costate, costate_trajectory[frame][0, :, 3])

        return line_state1, line_state2, line_state3, line_state4, line_lambda1, line_lambda2, line_lambda3,line_lambda4

    # Create animation
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=total_steps_sim, interval=100, blit=True)
    output_dir = f"./figs/animation_{option}.gif"
    ani.save(output_dir, writer=animation.PillowWriter(fps=20))
    print(f"Animation saved to {output_dir}")




def save_animation_unicycle_trajectory(x_robot, y_robot, theta_robot, initial_state_option, gif_name, obstacles=None,
                                       robot_r=0.25, margin=0.05):
    os.makedirs("./animation/unicycle", exist_ok=True)
    if initial_state_option == 'a':
       
        start_pos = [-1.9, 2.0, -1.79, 0.0]
    else:
        start_pos = [-5.24, 4.11, 2.72, 0.0]

    if initial_state_option == 'c':
        end_pos = [1, 1, 0, 0]
    else:
        end_pos = [0, 0, 0, 0]

    # Create a figure for the animation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(min(x_robot) - 1, max(x_robot) + 1)
    ax.set_ylim(min(y_robot) - 1, max(y_robot) + 1)
    ax.set_aspect('equal')  # Ensure equal scale for x and y axes
    ax.set_xlabel("X (m)", fontsize=20, fontweight='bold')
    ax.set_ylabel("Y (m)", fontsize=20, fontweight='bold')
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.3)
    

   # Draw obstacles and safety boundaries
    if obstacles is not None:
        for j, obs in enumerate(obstacles):
            xo = float(obs["xo"])
            yo = float(obs["yo"])
            r_obs = float(obs["r"])
            r_eff = r_obs + robot_r + margin
        # Physical obstacle
            ax.add_patch(plt.Circle((xo, yo), r_obs,fill=True, linewidth=3, color='black', label="_no_legend_" , zorder=4))

        # Safety boundary enforced by CBF/HOCBF
            ax.add_patch(plt.Circle((xo, yo), r_eff, fill=False, linewidth=2, linestyle='--', color='red', label="_no_legend_" , zorder=3))

    # Draw the robot components
    robot_body = plt.Circle((0, 0), 0.25, color='cyan', fill=True, linewidth=2)  # Robot circular base   #was 0.25
    wheel_width = 0.1
    wheel_height = 0.2

    wheel1 = plt.Rectangle((-wheel_height / 2, -wheel_width / 2), wheel_height, wheel_width, color='gold')  # Left wheel
    wheel2 = plt.Rectangle((-wheel_height / 2, -wheel_width / 2), wheel_height, wheel_width, color='gold')  # Right wheel

    # Add the robot components to the plot
    ax.add_patch(robot_body)
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)
    
    # if i want to include heading arrow 
   # front_len = 0.35
    #front_arrow, = ax.plot([], [], color='black', linewidth=3, solid_capstyle='round')


    # Add the black solid circles to represent obstacles
    start_xy = plt.Circle((start_pos[0], start_pos[1]), 0.2, color='green', fill=True, label="_no_legend_", zorder=5)
    goal_xy = plt.Circle((end_pos[0], end_pos[1]), 0.2, color='red', fill=True, label="_no_legend_", zorder=5)
    ax.add_patch(start_xy)
    ax.add_patch(goal_xy)
    # Add a dynamic trajectory line
    robot_path, = ax.plot([], [], linestyle="--", color="saddlebrown", label="Traveled Path", linewidth=2)
    ax.legend(loc="upper right",fontsize=20)

    # Animation function
    def update(frame):
        # Update the robot's position and orientation
        x, y, theta = x_robot[frame], y_robot[frame], theta_robot[frame]
        
        # Update robot body position
        robot_body.center = (x, y)
        
        # Wheel offsets relative to the robot's center
        wheel_offset = 0.3 # Distance of wheels from center  
        
        # Update left wheel position
        wheel1_center_x = x - wheel_offset * np.sin(theta)
        wheel1_center_y = y + wheel_offset * np.cos(theta)
        wheel1.set_xy((
            wheel1_center_x - wheel_height / 2 * np.cos(theta) + wheel_width / 2 * np.sin(theta),
            wheel1_center_y - wheel_height / 2 * np.sin(theta) - wheel_width / 2 * np.cos(theta),
        ))
        wheel1.angle = np.degrees(theta)
        
        # Update right wheel position
        wheel2_center_x = x + wheel_offset * np.sin(theta)
        wheel2_center_y = y - wheel_offset * np.cos(theta)
        wheel2.set_xy((
            wheel2_center_x - wheel_height / 2 * np.cos(theta) + wheel_width / 2 * np.sin(theta),
            wheel2_center_y - wheel_height / 2 * np.sin(theta) - wheel_width / 2 * np.cos(theta),
        ))
        wheel2.angle = np.degrees(theta)
        
        # Update the dynamic trajectory
        robot_path.set_data(x_robot[:frame], y_robot[:frame])  # Update only up to the current frame

        return robot_body, wheel1, wheel2, robot_path  
    # Create the animation
    time_per_step = 0.05 # in seconds, adjusted based on experiment results
    plt.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=len(x_robot), blit=True)
    fps = 1 / time_per_step
    if not animation.writers.is_available("ffmpeg"):
        try:
            import imageio_ffmpeg
            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

    if animation.writers.is_available("ffmpeg"):
        output_dir = f"./animation/unicycle/{gif_name}.mp4"
        anim.save(output_dir, writer="ffmpeg", fps=fps)
    else:
        output_dir = f"./animation/unicycle/{gif_name}.gif"
        anim.save(output_dir, writer=animation.PillowWriter(fps=fps))
    print(f"Animation (wheeled robot motion) saved to {output_dir}")


def save_unicycle_final_shot(x_robot, y_robot, file_name, start_xy=None, goal_xy=None, theta_robot=None, obstacles=None, robot_r=0.25, margin=0.05,
    show_safety_boundary=True, draw_robot=True, robot_alpha=0.45,):
    """
    Save a static snapshot of the executed trajectory with optional safety boundaries
    and a unicycle rendering at the final state.
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

    x_candidates = [float(np.min(x_robot)), float(np.max(x_robot)),start_marker_xy[0], goal_marker_xy[0],]
    y_candidates = [float(np.min(y_robot)), float(np.max(y_robot)), start_marker_xy[1], goal_marker_xy[1],]

    if obstacles is not None:
        for j, obs in enumerate(obstacles):
            xo = float(obs["xo"])
            yo = float(obs["yo"])
            r_obs = float(obs["r"])
            r_eff = r_obs + robot_r + margin


            ax.scatter( [xo], [yo], color="gray",s=45, alpha=0.5,label="Obstacle" if j == 0 else "_nolegend_", zorder=1,)
            ax.add_patch(plt.Circle((xo, yo),r_obs,color="gray",alpha=0.8,label="_nolegend_",zorder=1,))
            if show_safety_boundary:
                ax.add_patch(plt.Circle( (xo, yo), r_eff, fill=False, linestyle="--", linewidth=1.5, color="red", alpha=0.9,
                    label="_nolegend_" if j == 0 else "_nolegend_", zorder=2,))

            visible_r = r_eff if show_safety_boundary else r_obs
            x_candidates.extend([xo - visible_r, xo + visible_r])
            y_candidates.extend([yo - visible_r, yo + visible_r])

    ax.scatter([start_marker_xy[0]], [start_marker_xy[1]], color="green", s=45, alpha=0.5, label="Start", zorder=4)
    ax.scatter([goal_marker_xy[0]], [goal_marker_xy[1]], color="red", s=45, alpha=0.5, label="Goal", zorder=4)
    ax.add_patch(plt.Circle((start_marker_xy[0], start_marker_xy[1]), 0.15, color="green", fill=True,alpha=0.5, label="_nolegend_", zorder=4))
    ax.add_patch(plt.Circle((goal_marker_xy[0], goal_marker_xy[1]), 0.15, color="red", fill=True,alpha=0.5, label="_nolegend_", zorder=4))


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

        wheel_left = plt.Polygon(_wheel_polygon(wheel_left_center), closed=True,  color="gold", ec="black", lw=0.8,alpha=robot_alpha,zorder=6,)
        wheel_right = plt.Polygon(_wheel_polygon(wheel_right_center),closed=True, color="gold", ec="black", lw=0.8, alpha=robot_alpha, zorder=6,)
        ax.add_patch(wheel_left)
        ax.add_patch(wheel_right)


    pad = 0.35
    ax.set_xlim(min(x_candidates) - pad, max(x_candidates) + pad)
    ax.set_ylim(min(y_candidates) - pad, max(y_candidates) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)", fontsize=20, fontweight='bold')
    ax.set_ylabel("Y (m)", fontsize=20, fontweight='bold')
    ax.tick_params(axis="both", labelsize=24)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best", prop={"weight": "bold", "size": 20})
    plt.tight_layout()

    output_path = f"./figs/final_shot/{file_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Final-shot figure saved to {output_path}")
