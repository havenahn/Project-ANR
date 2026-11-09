import os
# Avoid MKL duplicate library crashes in some Anaconda setups.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from build_env import build_env
from HOCBF_SAC.dynamics import DynamicsModel
from HOCBF_SAC.unicycle_cbf_agent import UnicycleCBFAgent



def parse_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def resolve_resume_path(resume, env_name):
    if resume == "default":
        return os.path.join(os.getcwd(), "output", f"{env_name}-run0")
    if str(resume).isnumeric():
        return os.path.join(os.getcwd(), "output", f"{env_name}-run{resume}")
    return resume


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initial_distance_to_goal(state, goal):
    dx = float(state[0] - goal[0])
    dy = float(state[1] - goal[1])
    return float(np.hypot(dx, dy))


def sample_random_initial_states(num_trials, goal, d_min, bounds, rng, max_attempt_factor=200):
    accepted_states = []
    rejected_count = 0
    attempts = 0
    max_attempts = max_attempt_factor * max(num_trials, 1)

    while len(accepted_states) < num_trials and attempts < max_attempts:
        attempts += 1
        sample = np.array([rng.uniform(bounds["x"][0], bounds["x"][1]),rng.uniform(bounds["y"][0], bounds["y"][1]),
                rng.uniform(bounds["theta"][0], bounds["theta"][1]), rng.uniform(bounds["v"][0], bounds["v"][1]), ], dtype=float,)

        if initial_distance_to_goal(sample, goal) < d_min:
            rejected_count += 1
            continue

        accepted_states.append(sample)

    if len(accepted_states) < num_trials:
        raise RuntimeError( f"Could only sample {len(accepted_states)} states after {attempts} attempts. "
            "Reduce d_min or increase max_attempt_factor." )

    return accepted_states, rejected_count


def sample_grid_initial_states(num_trials, goal, d_min, bounds, rng, grid_n=10):
    if grid_n < 2:
        raise ValueError("grid_n must be >= 2 for grid sampling.")

    x_range = np.linspace(bounds["x"][0], bounds["x"][1], grid_n)
    y_range = np.linspace(bounds["y"][0], bounds["y"][1], grid_n)
    theta_range = np.linspace(bounds["theta"][0], bounds["theta"][1], grid_n)
    v_range = np.linspace(bounds["v"][0], bounds["v"][1], grid_n)

    x, y, theta, v = np.meshgrid(x_range, y_range, theta_range, v_range)
    all_grid_states = np.vstack([x.ravel(), y.ravel(), theta.ravel(), v.ravel()]).T
    d0_xy = np.linalg.norm(all_grid_states[:, :2] - goal[:2].reshape(1, 2), axis=1)
    valid_states = all_grid_states[d0_xy >= d_min]
    rejected_count = int(all_grid_states.shape[0] - valid_states.shape[0])

    if valid_states.shape[0] == 0:
        raise RuntimeError("No valid grid states satisfy d_min; reduce d_min or increase bounds.")

    # Randomly sample from grid states; use replacement only if requested trials exceed valid grid count.
    replace = num_trials > valid_states.shape[0]
    idx = rng.choice(valid_states.shape[0], size=num_trials, replace=replace)
    sampled = valid_states[idx]
    return [np.array(s, dtype=float) for s in sampled], rejected_count


def compute_msd(signal_1d, dt):
    signal_1d = np.asarray(signal_1d, dtype=float).reshape(-1)
    if signal_1d.size < 2:
        return 0.0
    signal_grad = np.gradient(signal_1d, dt)
    return float(np.mean(signal_grad ** 2))


def compute_rollout_metrics(state_traj, u_traj, goal, dt):
    x_traj = state_traj[:, 0]
    y_traj = state_traj[:, 1]
    theta_traj = state_traj[:, 2]
    v_traj = state_traj[:, 3]

    x_msd = compute_msd(x_traj, dt)
    y_msd = compute_msd(y_traj, dt)
    theta_msd = compute_msd(theta_traj, dt)
    v_msd = compute_msd(v_traj, dt)
    avg_state_msd = (x_msd + y_msd + theta_msd + v_msd) / 4.0

    if u_traj.shape[0] > 0:
        a_traj = u_traj[:, 0]
        w_traj = u_traj[:, 1]
        a_msd = compute_msd(a_traj, dt)
        w_msd = compute_msd(w_traj, dt)
        avg_control_msd = (a_msd + w_msd) / 2.0
    else:
        avg_control_msd = 0.0

    final_state = state_traj[-1]
    ace = float(np.sum(np.abs(final_state - goal)))
    return ace, avg_state_msd, avg_control_msd, final_state


def rollout_unicycle_no_obstacle(agent, dynamics_model, args, state_0):
    env = build_env(env_name=args.env_name, obs_config="none", rand_init=False, init_mode="manual", ref_x=args.zref_x,ref_y=args.zref_y, ref_theta=args.zref_theta,
        ref_v=args.zref_v, manual_init_state=state_0.tolist(),custom_obstacles=None, r_a=args.r_a, r_w=args.r_w,)
    
    if agent.cbf_layer:
        agent.cbf_layer.env = env

    observation, _ = env.reset()
    done = False

    state_traj = [np.copy(env.state)]
    action_traj = []

    while not done:
        action = agent.select_action(observation, dynamics_model, evaluate=True, safe_action=args.cbf_mode != "off",)[0]
        observation, _, done, _ = env.step(action)
        action_traj.append(np.copy(action))
        state_traj.append(np.copy(env.state))

    completion = bool(done and env.episode_step < env.max_episode_steps)
    dt = float(env.dt)
    env.close()

    return np.asarray(state_traj, dtype=float), np.asarray(action_traj, dtype=float), completion, dt


@dataclass
class TrialResult:
    trial: int
    initial_state: list
    initial_distance_xy: float
    success_ace: int
    success_goal_met: int
    ace: float
    state_msd: float
    control_msd: float
    final_state: list
    rollout_steps: int


def evaluate_randomized_initial_states_case_a(args):
    set_seed(args.seed)
    goal = np.array([args.zref_x, args.zref_y, args.zref_theta, args.zref_v], dtype=float)

    bounds = {"x": (args.init_x_min, args.init_x_max),"y": (args.init_y_min, args.init_y_max),"theta": (args.init_theta_min, args.init_theta_max),
        "v": (args.init_v_min, args.init_v_max),}

    base_env = build_env( env_name=args.env_name,obs_config="none",rand_init=False, init_mode="manual", ref_x=args.zref_x,ref_y=args.zref_y,
        ref_theta=args.zref_theta, ref_v=args.zref_v, custom_obstacles=None, r_a=args.r_a, r_w=args.r_w,)
    
    agent = UnicycleCBFAgent(base_env.observation_space.shape[0], base_env.action_space, base_env, args)
    dynamics_model = DynamicsModel(base_env, args)
    model_path = resolve_resume_path(args.resume, args.env_name)
    if not os.path.exists(os.path.join(model_path, "actor.pkl")):
        raise FileNotFoundError(f"Could not find actor weights at {model_path}/actor.pkl")
    if not os.path.exists(os.path.join(model_path, "critic.pkl")):
        raise FileNotFoundError(f"Could not find critic weights at {model_path}/critic.pkl")
    agent.load_weights(model_path)

    rng = np.random.default_rng(args.seed)
    if args.sample_mode == "uniform":
        initial_states, rejected_count = sample_random_initial_states(num_trials=args.num_trials, goal=goal,d_min=args.d_min, bounds=bounds, rng=rng,
            max_attempt_factor=args.max_attempt_factor,)
    else:
        initial_states, rejected_count = sample_grid_initial_states( num_trials=args.num_trials, goal=goal, d_min=args.d_min, bounds=bounds, rng=rng,grid_n=args.sample_grid_n,)

    os.makedirs(args.output_dir, exist_ok=True)

    trial_results = []
    failed_trials = []
    ace_all = []
    state_msd_all = []
    control_msd_all = []
    success_ace_all = []
    success_goal_all = []
    d0_all = []

    start_time = time.time()
    for trial_idx, state_0 in enumerate(initial_states, start=1):
        d0 = initial_distance_to_goal(state_0, goal)
        state_traj, action_traj, success_goal_met, dt = rollout_unicycle_no_obstacle(
            agent, dynamics_model, args, state_0
        )
        ace, state_msd, control_msd, final_state = compute_rollout_metrics(
            state_traj, action_traj, goal, dt
        )
        success_ace = int(ace <= args.success_threshold)

        trial = TrialResult(trial=trial_idx,initial_state=state_0.tolist(), initial_distance_xy=d0, success_ace=success_ace, success_goal_met=int(success_goal_met),
            ace=ace, state_msd=state_msd, control_msd=control_msd, final_state=final_state.tolist(),rollout_steps=int(action_traj.shape[0]),)
        trial_results.append(trial)
        if success_ace == 0:
            failed_trials.append(trial)

        d0_all.append(d0)
        ace_all.append(ace)
        state_msd_all.append(state_msd)
        control_msd_all.append(control_msd)
        success_ace_all.append(success_ace)
        success_goal_all.append(int(success_goal_met))
        if args.print_trial_logs:
            print(f"trial {trial_idx:03d}/{args.num_trials}: d0={d0:.3f}, "f"success_ace={success_ace}, goal_met={int(success_goal_met)}, ACE={ace:.4f}, "
                f"state_MSD={state_msd:.4f}, control_MSD={control_msd:.4f}, "
                  f"final_state=[{final_state[0]:+.2f}, {final_state[1]:+.2f}, {final_state[2]:+.2f}, {final_state[3]:+.2f}]")

    elapsed = time.time() - start_time

    ace_arr = np.asarray(ace_all, dtype=float)
    state_msd_arr = np.asarray(state_msd_all, dtype=float)
    control_msd_arr = np.asarray(control_msd_all, dtype=float)
    success_ace_arr = np.asarray(success_ace_all, dtype=float)
    success_goal_arr = np.asarray(success_goal_all, dtype=float)
    d0_arr = np.asarray(d0_all, dtype=float)

    aggregate = {"device": "cuda" if torch.cuda.is_available() and args.cuda else "cpu", "model_path": model_path,"seed": args.seed, "num_trials": args.num_trials,
        "goal": goal.tolist(), "d_min": args.d_min, "rejected_samples_due_to_d_min": int(rejected_count), "mean_initial_distance_xy": float(np.mean(d0_arr)),
        "success_threshold_ace": args.success_threshold, "success_rate_ace": float(np.mean(success_ace_arr)), "success_rate_goal_met": float(np.mean(success_goal_arr)),
        "average_ace": float(np.mean(ace_arr)),"average_state_msd": float(np.mean(state_msd_arr)), "average_control_msd": float(np.mean(control_msd_arr)),
        "elapsed_seconds": float(elapsed), "seconds_per_trial": float(elapsed / max(args.num_trials, 1)), }

    summary_name = args.summary_name
    if summary_name is None or summary_name.strip() == "":
        resume_tag = str(args.resume).replace(os.sep, "_").replace(":", "_")
        summary_name = f"case_a_randomized_resume_{resume_tag}_n{args.num_trials}_seed{args.seed}"

    summary_path = os.path.join(args.output_dir, f"{summary_name}.json")
    payload = {
        "config": {
            "env_name": args.env_name,
            "cbf_mode": args.cbf_mode,
            "obs_config": "none",
            "bounds": bounds,
            "sample_mode": args.sample_mode,
            "sample_grid_n": args.sample_grid_n,
            "ref_state": goal.tolist(),
            "max_attempt_factor": args.max_attempt_factor,
        },
        "aggregate": aggregate,
        "trials": [asdict(t) for t in trial_results],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Randomized Initial-State Evaluation (Case a, no obstacles) ===")
    print(f"Using device: {aggregate['device']}")
    print(f"model path: {aggregate['model_path']}")
    print(f"random seed: {aggregate['seed']}")
    print(f"trials: {aggregate['num_trials']}")
    print(f"goal: {aggregate['goal']}")
    print(f"minimum initial distance to goal (d_min): {aggregate['d_min']:.3f}")
    print(f"rejected samples due to d_min: {aggregate['rejected_samples_due_to_d_min']}")
    print(f"mean initial distance (accepted): {aggregate['mean_initial_distance_xy']:.4f}")
    print(f"success criterion (ACE): <= {aggregate['success_threshold_ace']:.3f}")
    print(f"success rate (ACE): {100.0 * aggregate['success_rate_ace']:.2f}%")
    #print(f"success rate (goal_met): {100.0 * aggregate['success_rate_goal_met']:.2f}%")
    print(f"average ACE: {aggregate['average_ace']:.4f}")
    print(f"average state MSD: {aggregate['average_state_msd']:.4f}")
    print(f"average control MSD: {aggregate['average_control_msd']:.4f}")
    print(
        f"elapsed: {aggregate['elapsed_seconds']:.2f}s "
        f"({aggregate['seconds_per_trial']:.3f}s/trial)"
    )
    print(
        f"failed trials (ACE > {aggregate['success_threshold_ace']:.3f}): "
        f"{len(failed_trials)}/{args.num_trials}"
    )
    if args.print_failed_initial_states:
        if len(failed_trials) == 0:
            print("No failed trials.")
        else:
            max_to_print = len(failed_trials) if args.max_failed_print <= 0 else min(
                len(failed_trials), args.max_failed_print
            )
            print(f"Initial states for failed trials (showing {max_to_print}/{len(failed_trials)}):")
            for trial in failed_trials[:max_to_print]:
                s = trial.initial_state
                print(
                    f"  trial {trial.trial:03d}: "
                    f"[{s[0]:+.3f}, {s[1]:+.3f}, {s[2]:+.3f}, {s[3]:+.3f}]"
                )
    print(f"summary json saved to: {summary_path}")

    base_env.close()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Randomized initial-state evaluation for case-a (no obstacles)."
    )

    # Experiment controls
    parser.add_argument("--env_name", default="Unicycle", type=str)
    parser.add_argument("--resume", default="1", type=str, help="Run id or model folder path.")
    parser.add_argument("--num_trials", default=100, type=int)
    parser.add_argument("--success_threshold", default=0.6, type=float)
    parser.add_argument("--d_min", default=0.5, type=float)
    parser.add_argument("--max_attempt_factor", default=200, type=int)
    parser.add_argument("--sample_mode", default="uniform", type=str, choices=["uniform", "grid"])
    parser.add_argument("--sample_grid_n", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output_dir", default="./figs/randomized_case_a", type=str)
    parser.add_argument("--summary_name", default="", type=str)
    parser.add_argument("--print_trial_logs", action="store_true", help="print per-trial metrics")
    parser.add_argument(
        "--print_failed_initial_states",
        type=parse_bool,
        default=True,
        help="print initial states for failed trials (based on ACE threshold)",
    )
    parser.add_argument("--max_failed_print", default=0, type=int, help="max failed initial states to print (<=0 means print all)", )

    # Training-domain bounds for randomized initial states.
    parser.add_argument("--init_x_min", default=-2.0, type=float)
    parser.add_argument("--init_x_max", default=2.0, type=float)
    parser.add_argument("--init_y_min", default=-2.0, type=float)
    parser.add_argument("--init_y_max", default=2.0, type=float)
    parser.add_argument("--init_theta_min", default=-2.0, type=float)
    parser.add_argument("--init_theta_max", default=2.0, type=float)
    parser.add_argument("--init_v_min", default=-2.0, type=float)
    parser.add_argument("--init_v_max", default=2.0, type=float)

    # Reference state (goal).
    parser.add_argument("--zref_x", default=0.0, type=float)
    parser.add_argument("--zref_y", default=0.0, type=float)
    parser.add_argument("--zref_theta", default=0.0, type=float)
    parser.add_argument("--zref_v", default=0.0, type=float)

    # Agent/model options needed to rebuild actor/critic for loading weights.
    parser.add_argument("--policy", default="Gaussian", type=str)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--automatic_entropy_tuning", type=parse_bool, default=True)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cbf_mode", default="off", type=str, choices=["off", "baseline"])

    # Cost and CBF parameters (used when cbf_mode=baseline).
    parser.add_argument("--r_a", default=1.0, type=float)
    parser.add_argument("--r_w", default=1.0, type=float)
    parser.add_argument("--robot_r", default=0.25, type=float)
    parser.add_argument("--safety_margin", default=0.05, type=float)
    parser.add_argument("--cbf_buffer", default=None, type=float)
    parser.add_argument("--k1", default=3.0, type=float)
    parser.add_argument("--k2", default=3.0, type=float)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.env_name != "Unicycle":
        raise ValueError("This script currently supports env_name='Unicycle' only.")
    if args.cbf_buffer is None:
        args.cbf_buffer = args.safety_margin
    if args.cuda:
        torch.cuda.set_device(args.device_num)

    evaluate_randomized_initial_states_case_a(args)
