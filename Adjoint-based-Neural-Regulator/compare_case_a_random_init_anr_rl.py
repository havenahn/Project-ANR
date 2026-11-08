import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
ANR_ROOT = ROOT / "ANR"
RL_ROOT = ROOT / "RL_SAC_HOCBF"

GOAL = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
TRAINING_DOMAIN = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "theta": (-2.0, 2.0), "v": (0.0, 0.0),}

SUCCESS_COLOR = "tab:blue"
FAILURE_COLOR = "tab:red"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def initial_distance_xy(state, goal):
    state = np.asarray(state, dtype=float).reshape(4,)
    goal = np.asarray(goal, dtype=float).reshape(4,)
    return float(np.hypot(state[0] - goal[0], state[1] - goal[1]))


def sample_shared_initial_states(num_trials, seed, d_min, bounds, max_attempt_factor):
    rng = np.random.default_rng(seed)
    states = []
    rejected_count = 0
    attempts = 0
    max_attempts = max_attempt_factor * max(num_trials, 1)

    while len(states) < num_trials and attempts < max_attempts:
        attempts += 1
        state = np.array([
                rng.uniform(bounds["x"][0], bounds["x"][1]),
                rng.uniform(bounds["y"][0], bounds["y"][1]),
                rng.uniform(bounds["theta"][0], bounds["theta"][1]),
                0.0,], dtype=float,)
        if initial_distance_xy(state, GOAL) < d_min:
            rejected_count += 1
            continue
        states.append(state)

    if len(states) < num_trials:
        raise RuntimeError(
            f"Could only sample {len(states)} states after {attempts} attempts. "
            "Reduce d_min or increase max_attempt_factor.")

    return np.asarray(states, dtype=float), rejected_count


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


def load_anr_config():
    import importlib.util

    spec = importlib.util.spec_from_file_location("case_a_compare_anr_config", str(ANR_ROOT / "config.py"))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load ANR config from {ANR_ROOT / 'config.py'}")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def build_anr_context(args):
    config = load_anr_config()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model = CoNN(int(args.anr_model_n)).to(device)
    model_name = f"anr_N{args.anr_model_n}_h{args.anr_model_h}_seed_{args.anr_model_seed}_e{args.anr_model_epoch}.pth"
    model_path = ANR_ROOT / "model" / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ANR weights: {model_path}")
    model.load_state_dict(safe_torch_load(model_path, device))
    model.eval()
    return {"config": config, "device": device, "model": model, "model_path": model_path}


def rollout_anr(context, initial_state):
    config = context["config"]
    state_k = np.asarray(initial_state, dtype=float).reshape(1, 4).copy()
    state_traj = [state_k.copy()]
    action_traj = []

    with torch.no_grad():
        for _ in range(int(config.total_steps_sim)):
            state_tensor = torch.tensor(state_k, dtype=torch.float32, device=context["device"])
            costate = context["model"](state_tensor).detach().cpu().numpy()

            a = float(np.clip(-0.5 * float(costate[0, 0, 3]), float(config.a_min), float(config.a_max)))
            omega = float(np.clip(-0.5 * float(costate[0, 0, 2]), float(config.w_min), float(config.w_max)))
            u_k = np.array([a, omega], dtype=float)

            action_traj.append(u_k.copy())
            state_k = rk4_step(state_k, u_k, float(config.dt))
            state_traj.append(state_k.copy())

    return {
        "state_traj": np.asarray(state_traj, dtype=float).squeeze(1),
        "action_traj": np.asarray(action_traj, dtype=float),
        "dt": float(config.dt),
    }


def build_rl_runner():
    if str(RL_ROOT) not in sys.path:
        sys.path.insert(0, str(RL_ROOT))
    from build_env import build_env
    from HOCBF_SAC.dynamics import DynamicsModel
    from HOCBF_SAC.unicycle_cbf_agent import UnicycleCBFAgent

    return build_env, DynamicsModel, UnicycleCBFAgent


def resolve_rl_model_path(resume, env_name):
    if str(resume).isnumeric():
        return RL_ROOT / "output" / f"{env_name}-run{resume}"
    return Path(resume).expanduser().resolve()


def build_rl_context(args):
    build_env, DynamicsModel, UnicycleCBFAgent = build_rl_runner()
    rl_args = SimpleNamespace(gamma=0.99, tau=0.005, alpha=0.2, policy="Gaussian",target_update_interval=1, automatic_entropy_tuning=True, hidden_size=256, 
        lr=0.0003, cuda=args.cuda, cbf_mode="off", robot_r=args.robot_r, safety_margin=args.safety_margin,    cbf_buffer=args.safety_margin, k1=3.0, k2=3.0,)
    
    base_env = build_env( env_name=args.env_name, obs_config="none", rand_init=False, init_mode="manual", ref_x=0.0, ref_y=0.0, ref_theta=0.0,
        ref_v=0.0, manual_init_state=GOAL.tolist(),   custom_obstacles=None, r_a=args.r_a, r_w=args.r_w, )
    
    agent = UnicycleCBFAgent(base_env.observation_space.shape[0], base_env.action_space, base_env, rl_args)
    dynamics_model = DynamicsModel(base_env, rl_args)
    model_path = resolve_rl_model_path(args.rl_resume, args.env_name)
    if not (model_path / "actor.pkl").exists():
        raise FileNotFoundError(f"Missing RL actor weights: {model_path / 'actor.pkl'}")
    if not (model_path / "critic.pkl").exists():
        raise FileNotFoundError(f"Missing RL critic weights: {model_path / 'critic.pkl'}")
    agent.load_weights(str(model_path))

    return {"build_env": build_env,"agent": agent, "dynamics_model": dynamics_model,"base_env": base_env, "model_path": model_path, "args": args,}


def rollout_rl(context, initial_state):
    args = context["args"]

    env = context["build_env"](env_name=args.env_name, obs_config="none", rand_init=False, init_mode="manual",ref_x=0.0,ref_y=0.0,ref_theta=0.0, ref_v=0.0,
        manual_init_state=np.asarray(initial_state, dtype=float).reshape(4,).tolist(),custom_obstacles=None, r_a=args.r_a,r_w=args.r_w,)

    if context["agent"].cbf_layer:
        context["agent"].cbf_layer.env = env

    observation, _ = env.reset()
    done = False
    state_traj = [np.copy(env.state)]
    action_traj = []

    while not done:
        action = context["agent"].select_action( observation,  context["dynamics_model"], evaluate=True, safe_action=False,)[0]
        observation, _, done, _ = env.step(action)
        action_traj.append(np.copy(action))
        state_traj.append(np.copy(env.state))

    dt = float(env.dt)
    env.close()
    return {"state_traj": np.asarray(state_traj, dtype=float), "action_traj": np.asarray(action_traj, dtype=float), "dt": dt,}


def compute_msd(signal_1d, dt):
    signal_1d = np.asarray(signal_1d, dtype=float).reshape(-1)
    if signal_1d.size < 2:
        return 0.0
    signal_grad = np.gradient(signal_1d, dt)
    return float(np.mean(signal_grad ** 2))


def compute_rollout_metrics(rollout, goal):
    state_traj = rollout["state_traj"]
    action_traj = rollout["action_traj"]
    dt = float(rollout["dt"])

    x_msd = compute_msd(state_traj[:, 0], dt)
    y_msd = compute_msd(state_traj[:, 1], dt)
    theta_msd = compute_msd(state_traj[:, 2], dt)
    v_msd = compute_msd(state_traj[:, 3], dt)
    avg_state_msd = (x_msd + y_msd + theta_msd + v_msd) / 4.0

    if action_traj.shape[0] > 0:
        a_msd = compute_msd(action_traj[:, 0], dt)
        w_msd = compute_msd(action_traj[:, 1], dt)
        avg_control_msd = (a_msd + w_msd) / 2.0
    else:
        avg_control_msd = 0.0

    final_state = np.asarray(state_traj[-1], dtype=float).reshape(4,)
    ace = float(np.sum(np.abs(final_state - goal)))
    return ace, avg_state_msd, avg_control_msd, final_state


def summarize_trial(trial_idx, initial_state, rollout, success_threshold):
    ace, state_msd, control_msd, final_state = compute_rollout_metrics(rollout, GOAL)
    return {
        "trial": int(trial_idx),
        "initial_state": np.asarray(initial_state, dtype=float).reshape(4,).tolist(),
        "initial_distance_xy": initial_distance_xy(initial_state, GOAL),
        "success": int(ace <= success_threshold),
        "ace": float(ace),
        "state_msd": float(state_msd),
        "control_msd": float(control_msd),
        "final_state": final_state.tolist(),
        "rollout_steps": int(rollout["action_traj"].shape[0]),
        "state_traj": rollout["state_traj"],
    }


def compact_trial_for_json(trial):
    return {
        "trial": trial["trial"],
        "initial_state": trial["initial_state"],
        "initial_distance_xy": trial["initial_distance_xy"],
        "success": trial["success"],
        "ace": trial["ace"],
        "state_msd": trial["state_msd"],
        "control_msd": trial["control_msd"],
        "final_state": trial["final_state"],
        "rollout_steps": trial["rollout_steps"],
    }


def aggregate_method(trials):
    return {
        "success_count": int(sum(trial["success"] for trial in trials)),
        "success_rate": float(np.mean([trial["success"] for trial in trials])),
        "average_ace": float(np.mean([trial["ace"] for trial in trials])),
        "average_state_msd": float(np.mean([trial["state_msd"] for trial in trials])),
        "average_control_msd": float(np.mean([trial["control_msd"] for trial in trials])),
        "trials": [compact_trial_for_json(trial) for trial in trials],
    }


def save_plot(initial_states, method_trials, output_dir, success_threshold):
    output_dir.mkdir(parents=True, exist_ok=True)
    x_vals = [TRAINING_DOMAIN["x"][0], TRAINING_DOMAIN["x"][1], GOAL[0]]
    y_vals = [TRAINING_DOMAIN["y"][0], TRAINING_DOMAIN["y"][1], GOAL[1]]
    for trials in method_trials.values():
        for trial in trials:
            x_vals.extend(trial["state_traj"][:, 0].tolist())
            y_vals.extend(trial["state_traj"][:, 1].tolist())

    x_lo, x_hi = min(x_vals) - 0.35, max(x_vals) + 0.35
    y_lo, y_hi = min(y_vals) - 0.35, max(y_vals) + 0.35
    span = max(x_hi - x_lo, y_hi - y_lo)
    x_mid, y_mid = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    x_lo, x_hi = x_mid - 0.5 * span, x_mid + 0.5 * span
    y_lo, y_hi = y_mid - 0.5 * span, y_mid + 0.5 * span

    legend_handles = [
        Line2D([0], [0], color=SUCCESS_COLOR, linewidth=3.0, label="Successful"),
        Line2D([0], [0], color=FAILURE_COLOR, linewidth=3.0, label="Unsuccessful"),
        Line2D([0], [0], color="black", linewidth=3.0, linestyle="--", label="Training Domain"),
        Line2D([0], [0], marker="o", color="forestgreen", markersize=10, linestyle="None", label="Goal"),
        Line2D([0], [0], marker="o", color="black", markersize=10, linestyle="None", label="Initial States"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, method_name in zip(axes, ["ANR", "RL"]):
        ax.add_patch(Rectangle((-2.0, -2.0), 4.0, 4.0, facecolor="0.7", edgecolor="none", alpha=0.10, zorder=0))
        ax.add_patch(Rectangle((-2.0, -2.0), 4.0, 4.0, fill=False, edgecolor="black", linewidth=2.0, linestyle="--", zorder=1))
        for trial in method_trials[method_name]:
            color = SUCCESS_COLOR if trial["success"] else FAILURE_COLOR
            ax.plot(trial["state_traj"][:, 0], trial["state_traj"][:, 1], color=color, linewidth=1.6, alpha=0.85, zorder=2)

        ax.scatter(initial_states[:, 0], initial_states[:, 1], s=12, color="black", alpha=0.45, zorder=3)
        ax.scatter([GOAL[0]], [GOAL[1]], marker="o", s=180, color="forestgreen", edgecolors="black", linewidths=0.6, zorder=4)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.grid(True, alpha=0.35)
        ax.set_title("CBLNN" if method_name == "ANR" else "RL", fontsize=18, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", labelsize=13)
        plt.setp(ax.get_xticklabels(), fontsize=13, fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontsize=13, fontweight="bold")

    axes[0].set_ylabel("Y (m)", fontsize=16, fontweight="bold")
    axes[1].set_ylabel("")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.82),
        ncol=len(legend_handles),
        frameon=True,
        prop={"size": 12, "weight": "bold"},
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    figure_path = output_dir / "case_a_random_init_anr_rl.png"
    plt.savefig(figure_path, dpi=300)
    plt.close(fig)
    return figure_path


def run_comparison(args):
    set_seed(args.seed)
    bounds = {
        "x": (args.init_x_min, args.init_x_max),
        "y": (args.init_y_min, args.init_y_max),
        "theta": (args.init_theta_min, args.init_theta_max),
        "v": (0.0, 0.0),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_states, rejected_count = sample_shared_initial_states(
        num_trials=args.num_trials,
        seed=args.seed,
        d_min=args.d_min,
        bounds=bounds,
        max_attempt_factor=args.max_attempt_factor,
    )

    anr_context = build_anr_context(args)
    rl_context = build_rl_context(args)
    method_trials = {"ANR": [], "RL": []}

    start_time = time.time()
    try:
        for trial_idx, initial_state in enumerate(initial_states, start=1):
            anr_rollout = rollout_anr(anr_context, initial_state)
            rl_rollout = rollout_rl(rl_context, initial_state)
            method_trials["ANR"].append(summarize_trial(trial_idx, initial_state, anr_rollout, args.success_threshold))
            method_trials["RL"].append(summarize_trial(trial_idx, initial_state, rl_rollout, args.success_threshold))

            if args.print_trial_logs:
                anr_trial = method_trials["ANR"][-1]
                rl_trial = method_trials["RL"][-1]
                print(
                    f"trial {trial_idx:03d}/{args.num_trials}: "
                    f"d0={anr_trial['initial_distance_xy']:.3f}, "
                    f"ANR success={anr_trial['success']} ACE={anr_trial['ace']:.4f}, "
                    f"RL success={rl_trial['success']} ACE={rl_trial['ace']:.4f}"
                )
    finally:
        rl_context["base_env"].close()

    elapsed = time.time() - start_time
    figure_path = save_plot(initial_states, method_trials, output_dir, args.success_threshold)
    payload = {
        "config": {
            "case": "a",
            "goal": GOAL.tolist(),
            "num_trials": int(args.num_trials),
            "seed": int(args.seed),
            "success_threshold_ace": float(args.success_threshold),
            "d_min": float(args.d_min),
            "bounds": {k: [float(v[0]), float(v[1])] for k, v in bounds.items()},
            "v_fixed": 0.0,
            "rejected_samples_due_to_d_min": int(rejected_count),
            "anr_model_path": str(anr_context["model_path"]),
            "rl_model_path": str(rl_context["model_path"]),
            "elapsed_seconds": float(elapsed),
            "seconds_per_trial_pair": float(elapsed / max(args.num_trials, 1)),
        },
        "initial_states": initial_states.tolist(),
        "methods": {
            "ANR": aggregate_method(method_trials["ANR"]),
            "RL": aggregate_method(method_trials["RL"]),
        },
    }

    json_path = output_dir / "case_a_random_init_anr_rl.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return figure_path, json_path, payload


def build_parser():
    parser = argparse.ArgumentParser(description="Fair Case A random-initial-state comparison for ANR and RL.")
    parser.add_argument("--num_trials", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--success_threshold", default=0.6, type=float)
    parser.add_argument("--d_min", default=0.5, type=float)
    parser.add_argument("--max_attempt_factor", default=200, type=int)
    parser.add_argument("--output_dir", default=str(ROOT / "comparison_outputs" / "case_a_random_init_anr_rl"), type=str)
    parser.add_argument("--print_trial_logs", action="store_true")

    parser.add_argument("--init_x_min", default=-2.0, type=float)
    parser.add_argument("--init_x_max", default=2.0, type=float)
    parser.add_argument("--init_y_min", default=-2.0, type=float)
    parser.add_argument("--init_y_max", default=2.0, type=float)
    parser.add_argument("--init_theta_min", default=-2.0, type=float)
    parser.add_argument("--init_theta_max", default=2.0, type=float)

    parser.add_argument("--anr_model_seed", default=0, type=int)
    parser.add_argument("--anr_model_n", default=30, type=int)
    parser.add_argument("--anr_model_h", default=50, type=int)
    parser.add_argument("--anr_model_epoch", default=50, type=int)

    parser.add_argument("--env_name", default="Unicycle", type=str)
    parser.add_argument("--rl_resume", default="1", type=str, help="RL run id or full model folder path.")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--r_a", default=1.0, type=float)
    parser.add_argument("--r_w", default=1.0, type=float)
    parser.add_argument("--robot_r", default=0.25, type=float)
    parser.add_argument("--safety_margin", default=0.05, type=float)
    return parser


def main():
    args = build_parser().parse_args()
    if args.env_name != "Unicycle":
        raise ValueError("This script currently supports env_name='Unicycle' only.")

    print("=== Fair Case A Random-Initial-State Comparison ===")
    print(f"Shared initial states: n={args.num_trials}, seed={args.seed}, v=0")
    print(f"d_min: {args.d_min:.3f}, success threshold: ACE <= {args.success_threshold:.3f}")
    figure_path, json_path, payload = run_comparison(args)

    for method_name in ["ANR", "RL"]:
        summary = payload["methods"][method_name]
        print(
            f"{method_name}: success={summary['success_count']}/{args.num_trials}, "
            f"success_rate={100.0 * summary['success_rate']:.2f}%, "
            f"average_ace={summary['average_ace']:.4f}, "
            f"average_state_MSD={summary['average_state_msd']:.4f}, "
            f"average_control_MSD={summary['average_control_msd']:.4f}"
        )

    print(f"Saved figure: {figure_path}")
    print(f"Saved json: {json_path}")


if __name__ == "__main__":
    main()
