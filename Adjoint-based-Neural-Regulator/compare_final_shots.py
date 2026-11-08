import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


ROOT = Path(__file__).resolve().parent
RL_ROOT = ROOT / "RL_SAC_HOCBF"
ANR_ROOT = ROOT / "ANR"


CASES = {"a": {"start": np.array([-1.9, 2.0, -1.79, 0.0], dtype=float), "goal": np.array([0.0, 0.0, 0.0, 0.0], dtype=float), "methods": ("ANR", "RL", "MPC"),},
    "b": {"start": np.array([-5.24, 4.11, 2.72, 0.0], dtype=float),"goal": np.array([0.0, 0.0, 0.0, 0.0], dtype=float), "methods": ("ANR", "RL", "MPC"),  },
    "c": {"start": np.array([-5.24, 4.11, 2.72, 0.0], dtype=float), "goal": np.array([1.0, 1.0, 0.0, 0.0], dtype=float), "methods": ("ANR", "MPC"), },}


METHOD_STYLES = { "ANR": {"color": "tab:blue"}, "RL": {"color": "tab:orange"}, "MPC": {"color": "tab:purple"},}

START_COLOR = "forestgreen"
GOAL_COLOR = "crimson"


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def unicycle_dynamics(x, u):
    theta = float(x[0, 2])
    v = float(x[0, 3])
    a = float(u[0])
    omega = float(u[1])
    return np.array([v * np.cos(theta), v * np.sin(theta), omega, a], dtype=float)


def rk4_step(x, u, dt):
    k1 = unicycle_dynamics(x, u)
    k2 = unicycle_dynamics(x + dt / 2.0 * k1, u)
    k3 = unicycle_dynamics(x + dt / 2.0 * k2, u)
    k4 = unicycle_dynamics(x + dt * k3, u)
    x_next = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[0, 2] = wrap_to_pi(x_next[0, 2])
    return x_next


class CoNN(torch.nn.Module):
    def __init__(self, prediction_horizon):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.fc1 = torch.nn.Linear(4, 128)
        self.fc2 = torch.nn.Linear(128, 512)
        self.fc3 = torch.nn.Linear(512, 128)
        self.fc4 = torch.nn.Linear(128, 4 * prediction_horizon)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(-1, self.prediction_horizon, 4)


def build_rl_runner():
    if str(RL_ROOT) not in sys.path:
        sys.path.insert(0, str(RL_ROOT))

    from build_env import build_env
    from HOCBF_SAC.dynamics import DynamicsModel
    from HOCBF_SAC.unicycle_cbf_agent import UnicycleCBFAgent

    return build_env, DynamicsModel, UnicycleCBFAgent


def run_rl_case(case_name: str, run_id: int):
    build_env, DynamicsModel, UnicycleCBFAgent = build_rl_runner()
    case = CASES[case_name]

    args = SimpleNamespace( gamma=0.99,  tau=0.005, alpha=0.2, policy="Gaussian", target_update_interval=1, automatic_entropy_tuning=True, hidden_size=256,
        lr=0.0003, cuda=False, cbf_mode="off",  robot_r=0.25, safety_margin=0.05,  cbf_buffer=0.05, k1=3.0, k2=3.0,)

    env = build_env("Unicycle", obs_config="none", rand_init=False,init_mode="manual", ref_x=float(case["goal"][0]), ref_y=float(case["goal"][1]),
        ref_theta=float(case["goal"][2]), ref_v=float(case["goal"][3]), manual_init_state=case["start"].tolist(), custom_obstacles=None, r_a=1.0, r_w=1.0,)
    agent = UnicycleCBFAgent(env.observation_space.shape[0], env.action_space, env, args)
    dynamics_model = DynamicsModel(env, args)

    model_dir = RL_ROOT / "output" / f"Unicycle-run{run_id}"
    if not (model_dir / "actor.pkl").exists():
        raise FileNotFoundError(f"Missing RL actor weights: {model_dir / 'actor.pkl'}")
    if not (model_dir / "critic.pkl").exists():
        raise FileNotFoundError(f"Missing RL critic weights: {model_dir / 'critic.pkl'}")
    agent.load_weights(str(model_dir))

    obs, _ = env.reset()
    state_traj = [np.copy(env.state)]
    action_traj = []
    done = False

    while not done:
        action = agent.select_action(
            obs,
            dynamics_model,
            evaluate=True,
            safe_action=False,
        )[0]
        obs, _, done, _ = env.step(action)
        action_traj.append(np.asarray(action, dtype=float))
        state_traj.append(np.copy(env.state))

    env.close()
    return {
        "state_traj": np.asarray(state_traj, dtype=float),
        "action_traj": np.asarray(action_traj, dtype=float),
    }


def run_anr_case(case_name: str):
    config = load_module("anr_config_compare", ANR_ROOT / "config.py")
    case = CASES[case_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CoNN(int(config.N)).to(device)
    model_path = ANR_ROOT / "model" / "anr_N30_h50_seed_0_e50.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ANR weights: {model_path}")
    state_dict = safe_torch_load(model_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    state_k = case["start"].reshape(1, 4).copy()
    goal = case["goal"]
    state_traj = [state_k.copy()]
    action_traj = []

    with torch.no_grad():
        for _ in range(int(config.total_steps_sim)):
            state_k_tensor = torch.tensor(state_k, dtype=torch.float32, device=device)

            if case_name == "c":
                theta_err = wrap_to_pi(state_k[0, 2] - goal[2])
                model_input_np = np.array([[state_k[0, 0] - goal[0], state_k[0, 1] - goal[1], theta_err, state_k[0, 3] - goal[3]]], dtype=np.float32,)
                model_input = torch.tensor(model_input_np, dtype=torch.float32, device=device)
            else:
                model_input = state_k_tensor

            costate = model(model_input).detach().cpu().numpy()
            lambda3 = float(costate[0, 0, 2])
            lambda4 = float(costate[0, 0, 3])

            a = float(np.clip(-0.5 * lambda4, float(config.a_min), float(config.a_max)))
            omega = float(np.clip(-0.5 * lambda3, float(config.w_min), float(config.w_max)))
            u_k = np.array([a, omega], dtype=float)

            action_traj.append(u_k.copy())
            state_k = rk4_step(state_k, u_k, float(config.dt))
            state_traj.append(state_k.copy())

    return {"state_traj": np.asarray(state_traj, dtype=float).squeeze(1), "action_traj": np.asarray(action_traj, dtype=float), }


def run_mpc_case(case_name: str):
    config = load_module("anr_config_compare_for_mpc", ANR_ROOT / "config.py")
    mpc_module = load_module("anr_mpc_controller_compare", ANR_ROOT / "mpc_controller.py")
    controller = mpc_module.MPC_HOCBF(float(config.dt),int(config.N), float(config.a_min), float(config.a_max), float(config.w_min), float(config.w_max),
        float(config.alpha0),float(config.alpha1), )

    case = CASES[case_name]
    x_k = np.asarray(case["start"], dtype=float).reshape(4,)
    x_goal = np.asarray(case["goal"], dtype=float).reshape(4,)

    state_traj = [x_k.copy()]
    action_traj = []
    for _ in range(int(config.total_steps_sim)):
        u_k = controller.get_control_input(x_k, x_goal, [])
        u_k = np.asarray(u_k, dtype=float).reshape(2,)
        action_traj.append(u_k.copy())
        x_k = controller.rk4(x_k, u_k).full().reshape(4,)
        x_k[2] = wrap_to_pi(x_k[2])
        state_traj.append(x_k.copy())

    return {"state_traj": np.asarray(state_traj, dtype=float), "action_traj": np.asarray(action_traj, dtype=float),  }


def compute_abs_convergence_error(state_traj: np.ndarray, goal: np.ndarray):
    final_state = np.asarray(state_traj[-1], dtype=float).reshape(4,)
    goal = np.asarray(goal, dtype=float).reshape(4,)
    theta_err = wrap_to_pi(final_state[2] - goal[2])
    return float(abs(final_state[0] - goal[0]) + abs(final_state[1] - goal[1])  + abs(theta_err) + abs(final_state[3] - goal[3]))


def draw_unicycle_footprint(ax, end_xy, end_theta, robot_r=0.2, robot_alpha=0.45, edge_color="black", zorder_base=5,):
    body = plt.Circle(end_xy, robot_r, color="cyan", ec=edge_color, lw=1.2, alpha=robot_alpha, zorder=zorder_base,  )
    ax.add_patch(body)

    wheel_length = 0.8 * robot_r
    wheel_width = 0.32 * robot_r
    wheel_offset = 1.2 * robot_r

    c = np.cos(end_theta)
    s = np.sin(end_theta)
    fwd = np.array([c, s], dtype=float)
    lat = np.array([-s, c], dtype=float)

    def wheel_polygon(center_xy):
        center = np.array(center_xy, dtype=float)
        p1 = center - 0.5 * wheel_length * fwd - 0.5 * wheel_width * lat
        p2 = center + 0.5 * wheel_length * fwd - 0.5 * wheel_width * lat
        p3 = center + 0.5 * wheel_length * fwd + 0.5 * wheel_width * lat
        p4 = center - 0.5 * wheel_length * fwd + 0.5 * wheel_width * lat
        return np.vstack([p1, p2, p3, p4])

    wheel_left_center = np.array(end_xy, dtype=float) + wheel_offset * lat
    wheel_right_center = np.array(end_xy, dtype=float) - wheel_offset * lat

    wheel_left = plt.Polygon( wheel_polygon(wheel_left_center),  closed=True,  color="gold", ec="black", lw=0.8, alpha=robot_alpha, zorder=zorder_base + 1,  )
    wheel_right = plt.Polygon( wheel_polygon(wheel_right_center), closed=True, color="gold", ec="black", lw=0.8, alpha=robot_alpha, zorder=zorder_base + 1,  )
    ax.add_patch(wheel_left)
    ax.add_patch(wheel_right)


def save_overlay_plot(case_name: str, rollout_results: dict, output_dir: Path):
    case = CASES[case_name]
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))

    x_candidates = [float(case["start"][0]), float(case["goal"][0])]
    y_candidates = [float(case["start"][1]), float(case["goal"][1])]
    marker_size = 17 if case_name == "a" else 38
    marker_radius = 0.05 if case_name == "a" else 0.12
    robot_radius = 0.125 if case_name == "a" else 0.18

    for idx, (method_name, result) in enumerate(rollout_results.items()):
        traj = result["state_traj"]
        style = METHOD_STYLES[method_name]
        x_traj = traj[:, 0]
        y_traj = traj[:, 1]
        theta_final = float(traj[-1, 2])

        ax.plot(x_traj, y_traj, linewidth=3.5, color=style["color"],  label=method_name,  zorder=3, )
        draw_unicycle_footprint( ax, (float(x_traj[-1]), float(y_traj[-1])), theta_final,  robot_r=robot_radius, robot_alpha=0.45,
            edge_color=style["color"], zorder_base=5 + 2 * idx, )

        x_candidates.extend(x_traj.tolist())
        y_candidates.extend(y_traj.tolist())

    ax.scatter([case["start"][0]], [case["start"][1]], color=START_COLOR, s=marker_size, alpha=0.5, label="Start", zorder=4, )
    ax.scatter([case["goal"][0]], [case["goal"][1]], color=GOAL_COLOR, s=marker_size, alpha=0.5, label="Goal",  zorder=4, )
    ax.add_patch( plt.Circle((float(case["start"][0]), float(case["start"][1])), marker_radius, color=START_COLOR, fill=True, alpha=0.5, label="_nolegend_", zorder=4, ))
    ax.add_patch(plt.Circle( (float(case["goal"][0]), float(case["goal"][1])), marker_radius,   color=GOAL_COLOR, fill=True, alpha=0.5, label="_nolegend_",  zorder=4, ))

    x_min = min(x_candidates)
    x_max = max(x_candidates)
    y_min = min(y_candidates)
    y_max = max(y_candidates)
    pad = 0.35
    x_lo = x_min - pad
    x_hi = x_max + pad
    y_lo = y_min - pad
    y_hi = y_max + pad


    # Make the data window square by expanding the smaller span.
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo
    target_span = max(x_span, y_span)
    x_mid = 0.5 * (x_lo + x_hi)
    y_mid = 0.5 * (y_lo + y_hi)

    x_lo = x_mid - 0.5 * target_span
    x_hi = x_mid + 0.5 * target_span
    y_lo = y_mid - 0.5 * target_span
    y_hi = y_mid + 0.5 * target_span

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")
    tick_step = 0.5 if case_name == "a" else 1.0
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.set_xlabel("X (m)", fontsize=25, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=25, fontweight="bold")
    ax.set_title(f"Case {case_name.upper()}", fontsize=30, fontweight="bold")
    ax.tick_params(axis="both", labelsize=25)
    plt.setp(ax.get_xticklabels(), fontsize=25, fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontsize=25, fontweight="bold")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best", prop={"weight": "bold", "size": 25})
    plt.tight_layout()

    output_path = output_dir / f"case_{case_name}_no_obs_overlay.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def run_case(case_name: str, rl_run: int, output_dir: Path):
    case = CASES[case_name]
    rollout_results = {}

    if "ANR" in case["methods"]:
        anr_result = run_anr_case(case_name)
        anr_result["ace"] = compute_abs_convergence_error(anr_result["state_traj"], case["goal"])
        rollout_results["ANR"] = anr_result

    if "RL" in case["methods"]:
        rl_result = run_rl_case(case_name, rl_run)
        rl_result["ace"] = compute_abs_convergence_error(rl_result["state_traj"], case["goal"])
        rollout_results["RL"] = rl_result

    if "MPC" in case["methods"]:
        mpc_result = run_mpc_case(case_name)
        mpc_result["ace"] = compute_abs_convergence_error(mpc_result["state_traj"], case["goal"])
        rollout_results["MPC"] = mpc_result

    saved_path = save_overlay_plot(case_name, rollout_results, output_dir)
    return rollout_results, saved_path


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone final-shot overlay comparison for ANR, RL, and MPC." )
    parser.add_argument("--cases", nargs="+", default=["a", "b", "c"], choices=sorted(CASES.keys()), help="Cases to compare.", )
    parser.add_argument("--rl_run",type=int, default=1, help="RL run number under RL_SAC_HOCBF/output/Unicycle-run<id>.",  )
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "comparison_outputs" / "final_shot_overlays"),
        help="Directory where overlay figures will be saved.",   )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    print(f"Saving overlays to: {output_dir}")
    print(f"Using RL checkpoint: Unicycle-run{args.rl_run}")

    for case_name in args.cases:
        print(f"\nRunning case {case_name.upper()}...")
        results, saved_path = run_case(case_name, args.rl_run, output_dir)
        for method_name, result in results.items():
            final_state = result["state_traj"][-1]
            print( f"  {method_name}: ACE={result['ace']:.4f}, " f"final_state=[{final_state[0]:+.3f}, {final_state[1]:+.3f}, "f"{final_state[2]:+.3f}, {final_state[3]:+.3f}]")
        print(f"  Saved: {saved_path}")


if __name__ == "__main__":
    main()
