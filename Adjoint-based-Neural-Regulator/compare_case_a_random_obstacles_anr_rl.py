import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
ANR_ROOT = ROOT / "ANR"
RL_ROOT = ROOT / "RL_SAC_HOCBF"

START_STATE = np.array([-1.9, 2.0, -1.79, 0.0], dtype=float)
GOAL = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
BASE_OBSTACLES = [
    {"xo": -0.5, "yo": 1.1, "r": 0.1},
    {"xo": -1.5, "yo": 1.2, "r": 0.1},
]
OBSTACLE_BOUNDS = {"x": (-2.0, -0.2), "y": (0.4, 1.9)}

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


def sample_random_obstacles(
    base_obstacles,
    bounds,
    start_xy,
    goal_xy,
    rng,
    robot_r=0.25,
    margin=0.05,
    min_sep=0.5,
    max_tries=2000,
    radius_jitter_frac=0.0,
):
    obstacles = []
    x_min, x_max = bounds["x"]
    y_min, y_max = bounds["y"]
    sx, sy = start_xy
    gx, gy = goal_xy

    for base in base_obstacles:
        base_r = float(base["r"])
        if radius_jitter_frac > 0.0:
            jitter = rng.uniform(-radius_jitter_frac, radius_jitter_frac)
            r = max(1e-6, base_r * (1.0 + jitter))
        else:
            r = base_r

        r_eff = r + robot_r + margin
        accepted = False
        for _ in range(max_tries):
            xo = float(rng.uniform(x_min, x_max))
            yo = float(rng.uniform(y_min, y_max))

            if np.hypot(xo - sx, yo - sy) <= (r_eff + min_sep):
                continue
            if np.hypot(xo - gx, yo - gy) <= (r_eff + min_sep):
                continue

            overlap = False
            for obs in obstacles:
                d = np.hypot(xo - obs["xo"], yo - obs["yo"])
                min_allowed = r + obs["r"] + 2.0 * (robot_r + margin) + 0.1
                if d <= min_allowed:
                    overlap = True
                    break
            if overlap:
                continue

            obstacles.append({"xo": xo, "yo": yo, "r": r})
            accepted = True
            break

        if not accepted:
            raise RuntimeError("Failed to sample non-overlapping random obstacles.")

    return obstacles


def sample_shared_obstacle_trials(args):
    rng = np.random.default_rng(args.seed)
    start_xy = (float(START_STATE[0]), float(START_STATE[1]))
    goal_xy = (float(GOAL[0]), float(GOAL[1]))
    obstacle_trials = []
    fallback_count = 0

    for _ in range(args.num_trials):
        try:
            obstacles = sample_random_obstacles(
                base_obstacles=BASE_OBSTACLES,
                bounds=OBSTACLE_BOUNDS,
                start_xy=start_xy,
                goal_xy=goal_xy,
                rng=rng,
                robot_r=args.robot_r,
                margin=args.cbf_buffer,
                min_sep=args.obstacle_min_sep,
                max_tries=args.max_obstacle_tries,
                radius_jitter_frac=args.radius_jitter_frac,
            )
            used_fallback = False
        except RuntimeError:
            if not args.fallback_to_base_obstacles:
                raise
            obstacles = [dict(obs) for obs in BASE_OBSTACLES]
            used_fallback = True
            fallback_count += 1

        obstacle_trials.append(
            {
                "obstacles": [dict(obs) for obs in obstacles],
                "fallback": int(used_fallback),
            }
        )

    return obstacle_trials, fallback_count


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


def load_anr_modules():
    if str(ANR_ROOT) not in sys.path:
        sys.path.insert(0, str(ANR_ROOT))
    import config
    from utils import rk4, solve_qp_hocbf

    return config, rk4, solve_qp_hocbf


def build_anr_context(args):
    config, rk4, solve_qp_hocbf = load_anr_modules()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model = CoNN(int(args.anr_model_n)).to(device)
    model_name = (
        f"anr_N{args.anr_model_n}_h{args.anr_model_h}_"
        f"seed_{args.anr_model_seed}_e{args.anr_model_epoch}.pth"
    )
    model_path = ANR_ROOT / "model" / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ANR weights: {model_path}")
    model.load_state_dict(safe_torch_load(model_path, device))
    model.eval()
    return {
        "config": config,
        "device": device,
        "model": model,
        "model_path": model_path,
        "rk4": rk4,
        "solve_qp_hocbf": solve_qp_hocbf,
        "args": args,
    }


def rollout_anr(context, obstacles):
    config = context["config"]
    args = context["args"]
    state_k = START_STATE.reshape(1, 4).copy()
    state_traj = [state_k.copy()]
    action_traj = []

    with torch.no_grad():
        for _ in range(int(config.total_steps_sim)):
            state_tensor = torch.tensor(state_k, dtype=torch.float32, device=context["device"])
            costate = context["model"](state_tensor).detach().cpu().numpy()

            a, omega, _, _ = context["solve_qp_hocbf"](
                lambda1=float(costate[0, 0, 0]),
                lambda2=float(costate[0, 0, 1]),
                lambda3=float(costate[0, 0, 2]),
                lambda4=float(costate[0, 0, 3]),
                v=float(state_tensor[0, 3].detach().cpu().numpy()),
                theta=float(state_tensor[0, 2].detach().cpu().numpy()),
                state_k=state_k,
                obstacles=obstacles,
                a_min=float(config.a_min),
                a_max=float(config.a_max),
                w_min=float(config.w_min),
                w_max=float(config.w_max),
                k0=float(args.k1),
                k1=float(args.k2),
                robot_r=float(args.robot_r),
                margin=float(args.cbf_buffer),
            )
            u_k = np.array([float(a), float(omega)], dtype=float)
            action_traj.append(u_k.copy())
            state_k = context["rk4"](state_k, u_k)
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
    rl_args = SimpleNamespace(
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        policy="Gaussian",
        target_update_interval=1,
        automatic_entropy_tuning=True,
        hidden_size=256,
        lr=0.0003,
        cuda=args.cuda,
        cbf_mode=args.cbf_mode,
        robot_r=args.robot_r,
        safety_margin=args.safety_margin,
        cbf_buffer=args.cbf_buffer,
        k1=args.k1,
        k2=args.k2,
    )

    base_env = build_env(
        env_name=args.env_name,
        obs_config="none",
        rand_init=False,
        init_mode="manual",
        ref_x=GOAL[0],
        ref_y=GOAL[1],
        ref_theta=GOAL[2],
        ref_v=GOAL[3],
        manual_init_state=START_STATE.tolist(),
        custom_obstacles=[(obs["xo"], obs["yo"], obs["r"]) for obs in BASE_OBSTACLES],
        r_a=args.r_a,
        r_w=args.r_w,
    )

    agent = UnicycleCBFAgent(base_env.observation_space.shape[0], base_env.action_space, base_env, rl_args)
    dynamics_model = DynamicsModel(base_env, rl_args)
    model_path = resolve_rl_model_path(args.rl_resume, args.env_name)
    if not (model_path / "actor.pkl").exists():
        raise FileNotFoundError(f"Missing RL actor weights: {model_path / 'actor.pkl'}")
    if not (model_path / "critic.pkl").exists():
        raise FileNotFoundError(f"Missing RL critic weights: {model_path / 'critic.pkl'}")
    agent.load_weights(str(model_path))

    return {
        "build_env": build_env,
        "agent": agent,
        "dynamics_model": dynamics_model,
        "base_env": base_env,
        "model_path": model_path,
        "args": args,
    }


def rollout_rl(context, obstacles):
    args = context["args"]
    custom_obstacles = [(obs["xo"], obs["yo"], obs["r"]) for obs in obstacles]
    env = context["build_env"](
        env_name=args.env_name,
        obs_config="none",
        rand_init=False,
        init_mode="manual",
        ref_x=GOAL[0],
        ref_y=GOAL[1],
        ref_theta=GOAL[2],
        ref_v=GOAL[3],
        manual_init_state=START_STATE.tolist(),
        custom_obstacles=custom_obstacles,
        r_a=args.r_a,
        r_w=args.r_w,
    )

    if context["agent"].cbf_layer:
        context["agent"].cbf_layer.env = env

    observation, _ = env.reset()
    done = False
    state_traj = [np.copy(env.state)]
    action_traj = []

    while not done:
        action = context["agent"].select_action(
            observation,
            context["dynamics_model"],
            evaluate=True,
            safe_action=args.cbf_mode != "off",
        )[0]
        observation, _, done, _ = env.step(action)
        action_traj.append(np.copy(action))
        state_traj.append(np.copy(env.state))

    dt = float(env.dt)
    goal_met = bool(done and env.episode_step < env.max_episode_steps)
    env.close()
    return {
        "state_traj": np.asarray(state_traj, dtype=float),
        "action_traj": np.asarray(action_traj, dtype=float),
        "dt": dt,
        "goal_met": int(goal_met),
    }


def compute_msd(signal_1d, dt):
    signal_1d = np.asarray(signal_1d, dtype=float).reshape(-1)
    if signal_1d.size < 2:
        return 0.0
    signal_grad = np.gradient(signal_1d, dt)
    return float(np.mean(signal_grad ** 2))


def compute_rollout_metrics(rollout):
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
    ace = float(np.sum(np.abs(final_state - GOAL)))
    return ace, avg_state_msd, avg_control_msd, final_state


def summarize_trial(trial_idx, rollout, obstacles, fallback, success_threshold):
    ace, state_msd, control_msd, final_state = compute_rollout_metrics(rollout)
    return {
        "trial": int(trial_idx),
        "start_state": START_STATE.tolist(),
        "success": int(ace <= success_threshold),
        "goal_met": int(rollout.get("goal_met", 0)),
        "ace": float(ace),
        "state_msd": float(state_msd),
        "control_msd": float(control_msd),
        "final_state": final_state.tolist(),
        "rollout_steps": int(rollout["action_traj"].shape[0]),
        "fallback": int(fallback),
        "obstacles": [dict(obs) for obs in obstacles],
        "state_traj": rollout["state_traj"],
    }


def compact_trial_for_json(trial):
    return {
        "trial": trial["trial"],
        "start_state": trial["start_state"],
        "success": trial["success"],
        "goal_met": trial["goal_met"],
        "ace": trial["ace"],
        "state_msd": trial["state_msd"],
        "control_msd": trial["control_msd"],
        "final_state": trial["final_state"],
        "rollout_steps": trial["rollout_steps"],
        "fallback": trial["fallback"],
        "obstacles": trial["obstacles"],
    }


def aggregate_method(trials, exclude_fallback):
    included = [trial for trial in trials if not (exclude_fallback and trial["fallback"])]
    if len(included) == 0:
        success_rate = average_ace = average_state_msd = average_control_msd = float("nan")
    else:
        success_rate = float(np.mean([trial["success"] for trial in included]))
        average_ace = float(np.mean([trial["ace"] for trial in included]))
        average_state_msd = float(np.mean([trial["state_msd"] for trial in included]))
        average_control_msd = float(np.mean([trial["control_msd"] for trial in included]))

    return {
        "included_trials_for_metrics": int(len(included)),
        "success_count": int(sum(trial["success"] for trial in included)),
        "success_rate": success_rate,
        "average_ace": average_ace,
        "average_state_msd": average_state_msd,
        "average_control_msd": average_control_msd,
        "trials": [compact_trial_for_json(trial) for trial in trials],
    }


def run_comparison(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    obstacle_trials, fallback_count = sample_shared_obstacle_trials(args)
    anr_context = build_anr_context(args)
    rl_context = build_rl_context(args)
    method_trials = {"ANR": [], "RL": []}

    start_time = time.time()
    try:
        for trial_idx, obstacle_trial in enumerate(obstacle_trials, start=1):
            obstacles = obstacle_trial["obstacles"]
            fallback = obstacle_trial["fallback"]
            anr_rollout = rollout_anr(anr_context, obstacles)
            rl_rollout = rollout_rl(rl_context, obstacles)

            method_trials["ANR"].append(
                summarize_trial(trial_idx, anr_rollout, obstacles, fallback, args.success_threshold)
            )
            method_trials["RL"].append(
                summarize_trial(trial_idx, rl_rollout, obstacles, fallback, args.success_threshold)
            )

            if args.print_trial_logs:
                anr_trial = method_trials["ANR"][-1]
                rl_trial = method_trials["RL"][-1]
                print(
                    f"trial {trial_idx:03d}/{args.num_trials}: "
                    f"fallback={fallback}, "
                    f"ANR success={anr_trial['success']} ACE={anr_trial['ace']:.4f}, "
                    f"RL success={rl_trial['success']} ACE={rl_trial['ace']:.4f}, "
                    f"obstacles={obstacles}"
                )
    finally:
        rl_context["base_env"].close()

    elapsed = time.time() - start_time
    payload = {
        "config": {
            "case": "a",
            "start_state": START_STATE.tolist(),
            "goal": GOAL.tolist(),
            "base_obstacles": BASE_OBSTACLES,
            "obstacle_bounds": OBSTACLE_BOUNDS,
            "num_trials": int(args.num_trials),
            "seed": int(args.seed),
            "success_threshold_ace": float(args.success_threshold),
            "robot_r": float(args.robot_r),
            "safety_margin": float(args.safety_margin),
            "cbf_buffer": float(args.cbf_buffer),
            "obstacle_min_sep": float(args.obstacle_min_sep),
            "max_obstacle_tries": int(args.max_obstacle_tries),
            "radius_jitter_frac": float(args.radius_jitter_frac),
            "fallback_to_base_obstacles": bool(args.fallback_to_base_obstacles),
            "exclude_fallback_from_metrics": bool(args.exclude_fallback_from_metrics),
            "fallback_trials": int(fallback_count),
            "anr_model_path": str(anr_context["model_path"]),
            "rl_model_path": str(rl_context["model_path"]),
            "rl_cbf_mode": args.cbf_mode,
            "elapsed_seconds": float(elapsed),
            "seconds_per_trial_pair": float(elapsed / max(args.num_trials, 1)),
        },
        "shared_obstacle_trials": obstacle_trials,
        "methods": {
            "ANR": aggregate_method(method_trials["ANR"], args.exclude_fallback_from_metrics),
            "RL": aggregate_method(method_trials["RL"], args.exclude_fallback_from_metrics),
        },
    }

    json_path = output_dir / "case_a_random_obstacles_anr_rl.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return json_path, payload


def parse_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(description="Fair Case A randomized-obstacle comparison for ANR and RL.")
    parser.add_argument("--num_trials", default=100, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--success_threshold", default=0.6, type=float)
    parser.add_argument("--output_dir", default=str(ROOT / "comparison_outputs" / "case_a_random_obstacles_anr_rl"), type=str)
    parser.add_argument("--print_trial_logs", action="store_true")

    parser.add_argument("--obstacle_min_sep", default=0.5, type=float)
    parser.add_argument("--max_obstacle_tries", default=2000, type=int)
    parser.add_argument("--radius_jitter_frac", default=0.0, type=float)
    parser.add_argument("--fallback_to_base_obstacles", type=parse_bool, default=True)
    parser.add_argument("--exclude_fallback_from_metrics", type=parse_bool, default=True)

    parser.add_argument("--anr_model_seed", default=0, type=int)
    parser.add_argument("--anr_model_n", default=30, type=int)
    parser.add_argument("--anr_model_h", default=50, type=int)
    parser.add_argument("--anr_model_epoch", default=50, type=int)

    parser.add_argument("--env_name", default="Unicycle", type=str)
    parser.add_argument("--rl_resume", default="1", type=str, help="RL run id or full model folder path.")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cbf_mode", default="baseline", type=str, choices=["off", "baseline"])
    parser.add_argument("--r_a", default=1.0, type=float)
    parser.add_argument("--r_w", default=1.0, type=float)
    parser.add_argument("--robot_r", default=0.25, type=float)
    parser.add_argument("--safety_margin", default=0.05, type=float)
    parser.add_argument("--cbf_buffer", default=None, type=float)
    parser.add_argument("--k1", default=3.0, type=float)
    parser.add_argument("--k2", default=3.0, type=float)
    return parser


def main():
    args = build_parser().parse_args()
    if args.env_name != "Unicycle":
        raise ValueError("This script currently supports env_name='Unicycle' only.")
    if args.cbf_buffer is None:
        args.cbf_buffer = args.safety_margin

    print("=== Fair Case A Randomized-Obstacle Comparison ===")
    print(f"Shared start state: {START_STATE.tolist()}")
    print(f"Shared obstacle trials: n={args.num_trials}, seed={args.seed}")
    print(f"success threshold: ACE <= {args.success_threshold:.3f}")
    json_path, payload = run_comparison(args)

    for method_name in ["ANR", "RL"]:
        summary = payload["methods"][method_name]
        included = summary["included_trials_for_metrics"]
        print(
            f"{method_name}: success={summary['success_count']}/{included}, "
            f"success_rate={100.0 * summary['success_rate']:.2f}%, "
            f"average_ace={summary['average_ace']:.4f}, "
            f"average_state_MSD={summary['average_state_msd']:.4f}, "
            f"average_control_MSD={summary['average_control_msd']:.4f}"
        )

    print(f"fallback obstacle trials: {payload['config']['fallback_trials']}/{args.num_trials}")
    print(f"Saved json: {json_path}")


if __name__ == "__main__":
    main()
