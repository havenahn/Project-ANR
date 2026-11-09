import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import time
import numpy as np
import casadi as ca

from config import dt, N, total_steps_sim, a_max, a_min, w_max, w_min, alpha0, alpha1
from mpc_controller import MPC_HOCBF
from sim_ANR_randomized_obstacles import get_case_setup, sample_random_obstacles
from utils import set_seed


BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)


def wrap_to_pi(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def rollout_mpc(state_0, goal, obstacles):
    controller = MPC_HOCBF(dt, N, a_min, a_max, w_min, w_max, alpha0, alpha1)

    x_k = np.asarray(state_0, dtype=float).reshape(4,)
    x_goal = np.asarray(goal, dtype=float).reshape(4,)

    for _ in range(total_steps_sim):
        u_k = controller.get_control_input(ca.DM(x_k), ca.DM(x_goal), obstacles)
        x_k = controller.rk4(ca.DM(x_k), ca.DM(u_k)).full().reshape(4,)
        x_k[2] = wrap_to_pi(x_k[2])

    final_state = x_k.copy()
    theta_err = wrap_to_pi(final_state[2] - x_goal[2])
    abs_convergence_err = (
        abs(final_state[0] - x_goal[0])
        + abs(final_state[1] - x_goal[1])
        + abs(theta_err)
        + abs(final_state[3] - x_goal[3])
    )
    return float(abs_convergence_err), final_state


def evaluate_randomized_obstacles(initial_state_option="a", num_trials=50, success_threshold=0.60, seed=0):
    set_seed(seed)

    state_0, goal, base_obstacles, bounds = get_case_setup(initial_state_option)
    start_xy = (float(state_0[0, 0]), float(state_0[0, 1]))
    goal_xy = (float(goal[0]), float(goal[1]))

    successes_all = []
    errors_all = []
    successes_non_fallback = []
    errors_non_fallback = []
    fallback_successes = []
    fallback_errors = []
    fallback_trials = 0
    failed_trials = []

    t0 = time.time()
    for trial in range(num_trials):
        try:
            obstacles = sample_random_obstacles(base_obstacles=base_obstacles,  bounds=bounds, start_xy=start_xy,  goal_xy=goal_xy,)
            used_fallback = False
        except RuntimeError:
            obstacles = [dict(obs) for obs in base_obstacles]
            used_fallback = True
            fallback_trials += 1

        err, final_state = rollout_mpc(state_0, goal, obstacles)
        success = int(err < success_threshold)

        successes_all.append(success)
        errors_all.append(err)

        if used_fallback:
            fallback_successes.append(success)
            fallback_errors.append(err)
        else:
            successes_non_fallback.append(success)
            errors_non_fallback.append(err)

        if not success:
            failed_trials.append({
                "trial": trial + 1,
                "abs_conv_err": float(err),
                "final_state": [
                    float(final_state[0]),
                    float(final_state[1]),
                    float(final_state[2]),
                    float(final_state[3]),
                ],
                "obstacles": [dict(obs) for obs in obstacles],
                "fallback": int(used_fallback),
            })

        print(
            f"trial {trial + 1:03d}/{num_trials}: success={success}, abs_conv_err={err:.4f}, "
            f"final_state=[{final_state[0]:+.2f}, {final_state[1]:+.2f}, {final_state[2]:+.2f}, {final_state[3]:+.2f}], "
            f"fallback={int(used_fallback)}, counted_for_success_rate={int(not used_fallback)}, obstacles={obstacles}"
        )

    elapsed = time.time() - t0

    successes_non_fallback_arr = np.array(successes_non_fallback, dtype=float)
    errors_non_fallback_arr = np.array(errors_non_fallback, dtype=float)
    non_fallback_trials = int(successes_non_fallback_arr.size)
    if non_fallback_trials > 0:
        success_rate = float(successes_non_fallback_arr.mean())
    else:
        success_rate = float("nan")

    print("\n=== Randomized-Obstacle MPC Evaluation ===")
    print(f"case: {initial_state_option}")
    print(f"trials: {num_trials}")
    print(f"success criterion: abs_convergence_error < {success_threshold:.2f}")
    if non_fallback_trials > 0:
        print(f"success rate (non-fallback trials only): {success_rate:.4f} ({100.0 * success_rate:.2f}%)")
        print(f"abs convergence error mean (non-fallback): {errors_non_fallback_arr.mean():.4f}")
    else:
        print("success rate (non-fallback trials only): n/a (no non-fallback trials)")
        print("abs convergence error mean (non-fallback): n/a")
    print(f"sampling fallback trials: {fallback_trials}/{num_trials}")
    print(f"non-fallback trials used for success rate: {non_fallback_trials}/{num_trials}")
    print(f"failed trials (all): {len(failed_trials)}/{num_trials}")
    print(f"elapsed: {elapsed:.2f}s ({elapsed / max(num_trials, 1):.2f}s/trial)")

    if failed_trials:
        print("\nFailed-trial details (trial, ACE, final_state, obstacles, fallback):")
        for item in failed_trials:
            print(
                f"  trial={item['trial']:03d}, ACE={item['abs_conv_err']:.4f}, "
                f"final_state={[round(v, 3) for v in item['final_state']]}, "
                f"obstacles={item['obstacles']}, fallback={item['fallback']}"
            )


if __name__ == "__main__":
    INITIAL_STATE_OPTION = "b"   # "a", "b", or "c"
    NUM_TRIALS = 100
    SUCCESS_THRESHOLD = 0.6
    SEED = 0

    evaluate_randomized_obstacles( initial_state_option=INITIAL_STATE_OPTION, num_trials=NUM_TRIALS, success_threshold=SUCCESS_THRESHOLD, seed=SEED,)
