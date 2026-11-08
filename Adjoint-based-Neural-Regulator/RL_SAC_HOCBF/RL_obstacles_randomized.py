import os
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


def get_case_setup(case_option):
    state_0a = np.array([-1.9, 2.0, -1.79, 0.0], dtype=float)
    state_0b = np.array([-5.24, 4.11, 2.72, 0.0], dtype=float)
    state_0c = np.array([-5.24, 4.11, 2.72, 0.0], dtype=float)

    if case_option == "a":
        start_state = state_0a
        goal_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        base_obstacles = [{"xo": -0.5, "yo": 1.1, "r": 0.1}, {"xo": -1.5, "yo": 1.2, "r": 0.1},]
        bounds = {"x": (-2.0, -0.2), "y": (0.4, 1.9)}
    elif case_option == "b":
        start_state = state_0b
        goal_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        base_obstacles = [{"xo": -4.0, "yo": 2.0, "r": 0.3}, {"xo": -1.5, "yo": 1.5, "r": 0.3}, {"xo": -2.5, "yo": 3.0, "r": 0.3}, ]
        bounds = {"x": (-4.24, -0.8), "y": (1.0, 3.11)}
    elif case_option == "c":
        start_state = state_0c
        goal_state = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
        base_obstacles = [ {"xo": -4.0, "yo": 2.0, "r": 0.5}, {"xo": -1.5, "yo": 1.5, "r": 0.5}, {"xo": -2.5, "yo": 3.0, "r": 0.5}, ]
        bounds = {"x": (-4.6, -0.6), "y": (2.2, 4.8)}
    else:
        raise ValueError("case_option must be one of: a, b, c")

    return start_state, goal_state, base_obstacles, bounds


def sample_random_obstacles( base_obstacles, bounds,start_xy, goal_xy, rng,robot_r=0.25, margin=0.05, min_sep=0.5, max_tries=2000, radius_jitter_frac=0.0,):
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

            d_start = float(np.hypot(xo - sx, yo - sy))
            d_goal = float(np.hypot(xo - gx, yo - gy))
            if d_start <= (r_eff + min_sep) or d_goal <= (r_eff + min_sep):
                continue

            overlap = False
            for obs in obstacles:
                d = float(np.hypot(xo - obs["xo"], yo - obs["yo"]))
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


def rollout_trial(agent, dynamics_model, args, start_state, goal_state, obstacles):
    custom_obstacles = [(obs["xo"], obs["yo"], obs["r"]) for obs in obstacles]
    env = build_env(env_name=args.env_name, obs_config="none", rand_init=False, init_mode="manual", ref_x=goal_state[0], ref_y=goal_state[1],
        ref_theta=goal_state[2],  ref_v=goal_state[3], manual_init_state=start_state.tolist(), custom_obstacles=custom_obstacles, r_a=args.r_a, r_w=args.r_w, )
    if agent.cbf_layer:
        agent.cbf_layer.env = env

    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(
            obs,
            dynamics_model,
            evaluate=True,
            safe_action=args.cbf_mode != "off",
        )[0]
        obs, _, done, _ = env.step(action)

    final_state = np.copy(env.state).astype(float)
    steps = int(env.episode_step)
    goal_met = bool(done and steps < env.max_episode_steps)
    env.close()

    ace = float(np.sum(np.abs(final_state - goal_state)))
    return ace, final_state, steps, goal_met


@dataclass
class ObstacleTrialResult:
    trial: int
    success: int
    goal_met: int
    abs_convergence_err: float
    final_state: list
    rollout_steps: int
    fallback: int
    obstacles: list


def evaluate_randomized_obstacles(args):
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    start_state, goal_state, base_obstacles, bounds = get_case_setup(args.case_option)
    start_xy = (float(start_state[0]), float(start_state[1]))
    goal_xy = (float(goal_state[0]), float(goal_state[1]))

    base_env = build_env(env_name=args.env_name, obs_config="none", rand_init=False, init_mode="manual", ref_x=goal_state[0], ref_y=goal_state[1], ref_theta=goal_state[2],
        ref_v=goal_state[3],  manual_init_state=start_state.tolist(), custom_obstacles=[(obs["xo"], obs["yo"], obs["r"]) for obs in base_obstacles], r_a=args.r_a, r_w=args.r_w,)
    
    agent = UnicycleCBFAgent(base_env.observation_space.shape[0], base_env.action_space, base_env, args)
    dynamics_model = DynamicsModel(base_env, args)

    model_path = resolve_resume_path(args.resume, args.env_name)
    if not os.path.exists(os.path.join(model_path, "actor.pkl")):
        raise FileNotFoundError(f"Could not find actor weights at {model_path}/actor.pkl")
    if not os.path.exists(os.path.join(model_path, "critic.pkl")):
        raise FileNotFoundError(f"Could not find critic weights at {model_path}/critic.pkl")
    agent.load_weights(model_path)

    results = []
    successes = []
    goal_mets = []
    errors = []
    fallback_trials = 0

    start_t = time.time()
    for trial_idx in range(1, args.num_trials + 1):
        used_fallback = False
        try:
            obstacles = sample_random_obstacles(base_obstacles=base_obstacles, bounds=bounds,start_xy=start_xy, goal_xy=goal_xy, rng=rng, robot_r=args.robot_r,
                margin=args.cbf_buffer, min_sep=args.obstacle_min_sep, max_tries=args.max_obstacle_tries, radius_jitter_frac=args.radius_jitter_frac, )
        except RuntimeError:
            if not args.fallback_to_base_obstacles:
                raise
            obstacles = [dict(obs) for obs in base_obstacles]
            used_fallback = True
            fallback_trials += 1

        ace, final_state, rollout_steps, goal_met = rollout_trial( agent, dynamics_model, args, start_state, goal_state, obstacles )
        success = int(ace <= args.success_threshold)

        result = ObstacleTrialResult( trial=trial_idx, success=success, goal_met=int(goal_met), abs_convergence_err=ace,
            final_state=final_state.tolist(), rollout_steps=rollout_steps, fallback=int(used_fallback), obstacles=[dict(obs) for obs in obstacles],   )
        results.append(result)

        successes.append(success)
        goal_mets.append(int(goal_met))
        errors.append(ace)

        if args.print_trial_logs:
            print(f"trial {trial_idx:03d}/{args.num_trials}: success={success}, goal_met={int(goal_met)}, "
                f"ACE={ace:.4f}, final_state=[{final_state[0]:+.2f}, {final_state[1]:+.2f}, "
                f"{final_state[2]:+.2f}, {final_state[3]:+.2f}], fallback={int(used_fallback)}" )

    elapsed = float(time.time() - start_t)
    successes_arr = np.asarray(successes, dtype=float)
    goal_mets_arr = np.asarray(goal_mets, dtype=float)
    errors_arr = np.asarray(errors, dtype=float)
    fallback_arr = np.asarray([r.fallback for r in results], dtype=int)

    if args.exclude_fallback_from_metrics:
        metric_mask = fallback_arr == 0
    else:
        metric_mask = np.ones_like(fallback_arr, dtype=bool)

    metric_count = int(np.sum(metric_mask))
    if metric_count > 0:
        metric_success_rate = float(np.mean(successes_arr[metric_mask]))
        metric_goal_met_rate = float(np.mean(goal_mets_arr[metric_mask]))
        metric_average_ace = float(np.mean(errors_arr[metric_mask]))
        metric_ace_std = float(np.std(errors_arr[metric_mask]))
    else:
        metric_success_rate = float("nan")
        metric_goal_met_rate = float("nan")
        metric_average_ace = float("nan")
        metric_ace_std = float("nan")

    overall_success_rate = float(np.mean(successes_arr))
    overall_goal_met_rate = float(np.mean(goal_mets_arr))
    overall_average_ace = float(np.mean(errors_arr))
    overall_ace_std = float(np.std(errors_arr))

    failed = [r for r in results if r.success == 0]
    failed_included = [r for r in results if r.success == 0 and (r.fallback == 0 or not args.exclude_fallback_from_metrics)]

    aggregate = {"device": "cuda" if torch.cuda.is_available() and args.cuda else "cpu", "model_path": model_path, "case_option": args.case_option,
        "seed": args.seed, "num_trials": args.num_trials, "success_threshold_ace": args.success_threshold,
        "exclude_fallback_from_metrics": bool(args.exclude_fallback_from_metrics),
        "included_trials_for_metrics": metric_count,
        "success_rate": metric_success_rate,
        "goal_met_rate": metric_goal_met_rate,
        "average_ace": metric_average_ace,
        "ace_std": metric_ace_std,
        "overall_success_rate": overall_success_rate,
        "overall_goal_met_rate": overall_goal_met_rate,
        "overall_average_ace": overall_average_ace,
        "overall_ace_std": overall_ace_std,
        "fallback_trials": int(fallback_trials),
        "failed_trials": int(len(failed_included)),
        "failed_trials_overall": int(len(failed)),
        "elapsed_seconds": elapsed,
        "seconds_per_trial": float(elapsed / max(args.num_trials, 1)),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    summary_name = args.summary_name
    if summary_name is None or summary_name.strip() == "":
        summary_name = (
            f"obstacle_random_case_{args.case_option}_resume_{args.resume}_"
            f"n{args.num_trials}_seed{args.seed}"
        )
    summary_path = os.path.join(args.output_dir, f"{summary_name}.json")
    payload = {
        "config": {
            "env_name": args.env_name,
            "cbf_mode": args.cbf_mode,
            "start_state": start_state.tolist(),
            "goal_state": goal_state.tolist(),
            "base_obstacles": base_obstacles,
            "bounds": bounds,
            "robot_r": args.robot_r,
            "cbf_buffer": args.cbf_buffer,
            "obstacle_min_sep": args.obstacle_min_sep,
            "max_obstacle_tries": args.max_obstacle_tries,
            "radius_jitter_frac": args.radius_jitter_frac,
            "fallback_to_base_obstacles": args.fallback_to_base_obstacles,
        },
        "aggregate": aggregate,
        "trials": [asdict(r) for r in results],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    
    print(f"model path: {model_path}")
    print("\n=== Randomized-Obstacle RL Evaluation ===")
    print(f"case: {args.case_option}")
    print(f"trials: {args.num_trials}")
    print(
        "metrics trials: "
        f"{metric_count}/{args.num_trials} "
        f"(exclude fallback: {args.exclude_fallback_from_metrics})"
    )
    print(f"success criterion: ACE <= {args.success_threshold:.3f}")
    print(f"success rate: {100.0 * aggregate['success_rate']:.2f}%")
    #print(f"overall success rate (including fallback): {100.0 * aggregate['overall_success_rate']:.2f}%")
   # print(f"goal_met rate: {100.0 * aggregate['goal_met_rate']:.2f}%")
    print(f"average ACE: {aggregate['average_ace']:.4f}")
    #print(f"overall average ACE (including fallback): {aggregate['overall_average_ace']:.4f}")
    print(f"fallback trials: {aggregate['fallback_trials']}/{args.num_trials}")
    print(f"failed trials : {aggregate['failed_trials']}/{metric_count if metric_count > 0 else 0}")
    #print(f"failed trials overall: {aggregate['failed_trials_overall']}/{args.num_trials}")
    print(
        f"elapsed: {aggregate['elapsed_seconds']:.2f}s "
        f"({aggregate['seconds_per_trial']:.3f}s/trial)"
    )

    if args.print_failed_details:
        failed_to_print = failed_included
        if len(failed_to_print) == 0:
            print("No failed trials.")
        else:
            max_to_print = len(failed_to_print) if args.max_failed_print <= 0 else min(
                len(failed_to_print), args.max_failed_print
            )
            print(f"Failed-trial details (showing {max_to_print}/{len(failed_to_print)}):")
            for item in failed_to_print[:max_to_print]:
                fs = item.final_state
                print(f"  trial={item.trial:03d}, ACE={item.abs_convergence_err:.4f}, "
                    f"final_state=[{fs[0]:+.3f}, {fs[1]:+.3f}, {fs[2]:+.3f}, {fs[3]:+.3f}], "
                    f"fallback={item.fallback}, obstacles={item.obstacles}" )
                
    print(f"summary json saved to: {summary_path}")
    base_env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Randomized-obstacle evaluation for fixed case setup (a/b/c)." )

    # Experiment controls
    parser.add_argument("--env_name", default="Unicycle", type=str)
    parser.add_argument("--resume", default="1", type=str, help="Run id or model folder path.")
    parser.add_argument("--case_option", default="a", type=str, choices=["a", "b", "c"])
    parser.add_argument("--num_trials", default=100, type=int)
    parser.add_argument("--success_threshold", default=0.6, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output_dir", default="./figs/randomized_obstacles", type=str)
    parser.add_argument("--summary_name", default="", type=str)
    parser.add_argument("--print_trial_logs", action="store_true", help="print per-trial logs")
    parser.add_argument( "--print_failed_details",  type=parse_bool, default=True, help="print failed-trial details after summary",  )
    parser.add_argument( "--max_failed_print",  default=0,  type=int,  help="max failed trials to print (<=0 means print all)", )

    # Obstacle randomization controls
    parser.add_argument("--robot_r", default=0.25, type=float)
    parser.add_argument("--safety_margin", default=0.05, type=float)
    parser.add_argument("--cbf_buffer", default=None, type=float)
    parser.add_argument("--obstacle_min_sep", default=0.5, type=float)
    parser.add_argument("--max_obstacle_tries", default=2000, type=int)
    parser.add_argument("--radius_jitter_frac", default=0.0, type=float)
    parser.add_argument("--fallback_to_base_obstacles", type=parse_bool, default=True)
    parser.add_argument(
        "--exclude_fallback_from_metrics",
        type=parse_bool,
        default=True,
        help="if True, fallback trials are excluded from success/ACE summary metrics",
    )

    # Agent/model options to rebuild and load SAC policy.
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
    parser.add_argument("--cbf_mode", default="baseline", type=str, choices=["off", "baseline"])

    # Cost and HOCBF parameters
    parser.add_argument("--r_a", default=1.0, type=float)
    parser.add_argument("--r_w", default=1.0, type=float)
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

    evaluate_randomized_obstacles(args)
