import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import time
import numpy as np
import torch
import torch.nn as nn

from config import T_sim, total_steps_sim, a_max, a_min, w_max, w_min
from utils import solve_qp_hocbf, rk4, set_seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent


class CoNN(nn.Module):
    def __init__(self, prediction_horizon):
        super(CoNN, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 4 * prediction_horizon)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, self.prediction_horizon, 4)
        return x


def get_case_setup(initial_state_option):
    state_0a = np.array([[-1.9, 2.0,  -1.79,  0.0]])
    state_0b = np.array([[-5.24, 4.11, 2.72, 0.0]])
    state_0c = state_0b.copy()

    if initial_state_option == "a":
        state_0 = state_0a
        goal = np.array([0.0, 0.0, 0.0, 0.0])
    
        base_obstacles = [{"xo": -0.5, "yo":1.1, "r": 0.1},{"xo": -1.5, "yo":1.2, "r": 0.1},]    # for case a with 2 obstacles 
        bounds = {"x": (-2.0, -0.2), "y": (0.4, 1.9)}
    elif initial_state_option == "b":
        state_0 = state_0b
        goal = np.array([0.0, 0.0, 0.0, 0.0])
        base_obstacles = [{"xo": -4, "yo":2, "r": 0.3}, {"xo": -1.5, "yo":1.5, "r": 0.3}, {"xo": -2.5, "yo":3.0, "r": 0.3}]
        bounds = {"x": (-4.24, -0.8), "y": (1.0, 3.11)}
    else:
        state_0 = state_0c
        goal = np.array([1.0, 1.0, 0.0, 0.0])
    
        base_obstacles = [{"xo": -4, "yo":2, "r": 0.5}, {"xo": -1.5, "yo":1.5, "r": 0.5}, {"xo": -2.5, "yo":3.0, "r": 0.5}]
        bounds = {"x": (-4.6, -0.6), "y": (2.2, 4.8)}

    return state_0, goal, base_obstacles, bounds


def sample_random_obstacles(base_obstacles,bounds,start_xy,goal_xy, robot_r=0.25, margin=0.05,min_sep=0.5, max_tries=2000,):
    obstacles = []
    x_min, x_max = bounds["x"]
    y_min, y_max = bounds["y"]
    sx, sy = start_xy
    gx, gy = goal_xy

    for base in base_obstacles:
        r = float(base["r"])
        r_eff = r + robot_r + margin
        accepted = False

        for _ in range(max_tries):
            xo = np.random.uniform(x_min, x_max)
            yo = np.random.uniform(y_min, y_max)

            d_start = np.hypot(xo - sx, yo - sy)
            d_goal = np.hypot(xo - gx, yo - gy)
            if d_start <= (r_eff + min_sep) or d_goal <= (r_eff + min_sep):
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

            obstacles.append({"xo": float(xo), "yo": float(yo), "r": r})
            accepted = True
            break

        if not accepted:
            raise RuntimeError("Failed to sample non-overlapping random obstacles.")

    return obstacles


def rollout_anr(model, state_0, goal, obstacles, initial_state_option):
    x_ref, y_ref, theta_ref, v_ref = goal
    state_k = state_0.copy()
    state_traj = [state_k.copy()]

    for _ in range(total_steps_sim):
        state_k_tensor = torch.tensor(state_k, dtype=torch.float32, device=device)

        if initial_state_option == "c":
            theta_err = (state_k[0, 2] - theta_ref + np.pi) % (2.0 * np.pi) - np.pi
            error_k = np.array([[state_k[0, 0] - x_ref, state_k[0, 1] - y_ref, theta_err, state_k[0, 3] - v_ref]], dtype=np.float32,)
            model_input = torch.tensor(error_k, dtype=torch.float32, device=device)
        else:
            model_input = state_k_tensor

        costate_traj_k_hat = model(model_input).detach().cpu().numpy()
        lambda1_k_hat = float(costate_traj_k_hat[0, 0, 0])
        lambda2_k_hat = float(costate_traj_k_hat[0, 0, 1])
        lambda3_k_hat = float(costate_traj_k_hat[0, 0, 2])
        lambda4_k_hat = float(costate_traj_k_hat[0, 0, 3])

        a, omega, _, _ = solve_qp_hocbf(lambda1=lambda1_k_hat,lambda2=lambda2_k_hat,lambda3=lambda3_k_hat,lambda4=lambda4_k_hat,v=state_k_tensor[0, 3].detach().cpu().numpy(),
            theta=state_k_tensor[0, 2].detach().cpu().numpy(), state_k=state_k, obstacles=obstacles, a_max=a_max, a_min=a_min, w_max=w_max, w_min=w_min, )

        u_k = np.array([float(a), float(omega)], dtype=float)
        state_k = rk4(state_k, u_k)
        state_traj.append(state_k.copy())

    state_traj = np.array(state_traj).squeeze(1)
    final_state = state_traj[-1]

    if initial_state_option == "c":
        abs_convergence_err = (abs(final_state[0] - x_ref) + abs(final_state[1] - y_ref) + abs(final_state[2] - theta_ref) + abs(final_state[3] - v_ref)  )
    else:
        abs_convergence_err = (abs(final_state[0]) + abs(final_state[1]) + abs(final_state[2]) + abs(final_state[3]))

    return float(abs_convergence_err), final_state


def evaluate_randomized_obstacles(initial_state_option="a", num_trials=50, success_threshold=0.60, seed=0, model_n=30, model_h=50, model_epoch=50,):
    set_seed(seed)

    model = CoNN(model_n).to(device)
    model_name = f"anr_N{model_n}_h{model_h}_seed_{seed}_e{model_epoch}.pth"
    model_path = BASE_DIR / "model" / model_name
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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
            obstacles = sample_random_obstacles(base_obstacles=base_obstacles, bounds=bounds, start_xy=start_xy, goal_xy=goal_xy,   )
            used_fallback = False
        except RuntimeError:
            # Keep the sweep running even when random placement is infeasible.
            obstacles = [dict(obs) for obs in base_obstacles]
            used_fallback = True
            fallback_trials += 1

        err, final_state = rollout_anr(model, state_0, goal, obstacles, initial_state_option)
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
            failed_trials.append({ "trial": trial + 1,"abs_conv_err": float(err),"final_state": [float(final_state[0]), float(final_state[1]), float(final_state[2]), 
                            float(final_state[3])], "obstacles": [dict(obs) for obs in obstacles], "fallback": int(used_fallback), })

        print(f"trial {trial + 1:03d}/{num_trials}: success={success}, abs_conv_err={err:.4f}, "
            f"final_state=[{final_state[0]:+.2f}, {final_state[1]:+.2f}, {final_state[2]:+.2f}, {final_state[3]:+.2f}], "
            f"fallback={int(used_fallback)}, counted_for_success_rate={int(not used_fallback)}, obstacles={obstacles}" )

    elapsed = time.time() - t0

    successes_all_arr = np.array(successes_all, dtype=float)
    errors_all_arr = np.array(errors_all, dtype=float)
    successes_non_fallback_arr = np.array(successes_non_fallback, dtype=float)
    errors_non_fallback_arr = np.array(errors_non_fallback, dtype=float)
    fallback_successes_arr = np.array(fallback_successes, dtype=float)
    fallback_errors_arr = np.array(fallback_errors, dtype=float)

    non_fallback_trials = int(successes_non_fallback_arr.size)
    if non_fallback_trials > 0:
        success_rate = float(successes_non_fallback_arr.mean())
        success_std_pop = float(successes_non_fallback_arr.std(ddof=0))
        success_std_sample = (float(successes_non_fallback_arr.std(ddof=1)) if non_fallback_trials > 1 else 0.0  )
    else:
        success_rate = float("nan")
        success_std_pop = float("nan")
        success_std_sample = float("nan")

    print("\n=== Randomized-Obstacle ANR Evaluation ===")
    print(f"case: {initial_state_option}")
    print(f"trials: {num_trials}")
    print(f"success criterion: abs_convergence_error < {success_threshold:.2f}")
    print("success rate (non-fallback trials only): "f"{success_rate:.4f} ({100.0 * success_rate:.2f}%)"
        if non_fallback_trials > 0
        else "success rate (non-fallback trials only): n/a (no non-fallback trials)" )
    
    #print(f"success std (population): {success_std_pop:.4f}")
    print(f"sampling fallback trials: {fallback_trials}/{num_trials}")
    print(f"non-fallback trials used for success rate: {non_fallback_trials}/{num_trials}")
    print(f"failed trials (all): {len(failed_trials)}/{num_trials}")
    #print(f"success rate (all trials, including fallback): {successes_all_arr.mean():.4f} ({100.0 * successes_all_arr.mean():.2f}%)")
   # if fallback_trials > 0:
    #    print(
       #     "success rate (fallback-only): "
        #    f"{fallback_successes_arr.mean():.4f} ({100.0 * fallback_successes_arr.mean():.2f}%)"
      #  )
    print(f"abs convergence error mean (non-fallback): {errors_non_fallback_arr.mean():.4f}" if non_fallback_trials > 0 else "abs convergence error mean (non-fallback): n/a")
    #print(f"abs convergence error mean (all): {errors_all_arr.mean():.4f}")
   # print(f"abs convergence error std: {errors_arr.std(ddof=0):.4f}")
    #print(f"elapsed: {elapsed:.2f}s ({elapsed / max(num_trials, 1):.2f}s/trial)")

    if failed_trials:
        print("\nFailed-trial details (trial, ACE, final_state, obstacles, fallback):")
        for item in failed_trials:
            print(f"  trial={item['trial']:03d}, ACE={item['abs_conv_err']:.4f}, "f"final_state={[round(v, 3) for v in item['final_state']]}, "
                f"obstacles={item['obstacles']}, fallback={item['fallback']}")


if __name__ == "__main__":
    # Edit these settings as needed.
    INITIAL_STATE_OPTION = "b"   # "a", "b", or "c"
    NUM_TRIALS = 100
    SUCCESS_THRESHOLD = 0.6
    SEED = 0

    evaluate_randomized_obstacles(initial_state_option=INITIAL_STATE_OPTION,num_trials=NUM_TRIALS,success_threshold=SUCCESS_THRESHOLD, seed=SEED, 
                                  model_n=30, model_h=50, model_epoch=50,)
