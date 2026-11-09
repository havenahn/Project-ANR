import argparse
import os
import time

import numpy as np
import torch

from build_env import build_env
from HOCBF_SAC.dynamics import DynamicsModel
from HOCBF_SAC.replay_memory import ReplayMemory
from HOCBF_SAC.unicycle_cbf_agent import UnicycleCBFAgent
from HOCBF_SAC.utils import get_output_folder, prGreen, prYellow


def parse_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_obs_circles(spec):
    if spec is None or str(spec).strip() == "":
        return None
    # PowerShell users may pass `--obs_circles =...`; normalize that input.
    spec = str(spec).strip()
    if spec.startswith("="):
        spec = spec[1:].strip()
    circles = []
    for item in spec.split(";"):
        item = item.strip().strip('"').strip("'")
        if item.startswith("="):
            item = item[1:].strip()
        vals = [v.strip().strip('"').strip("'") for v in item.split(",")]
        if vals:
            vals[0] = vals[0].lstrip("=").strip()
        if len(vals) != 3:
            raise argparse.ArgumentTypeError("Each obstacle must be 'x,y,r'. Example: --obs_circles \"0,0,0.6;1.5,1.0,0.4\""  )
        try:
            x, y, r = map(float, vals)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Could not parse --obs_circles. Use: --obs_circles \"x1,y1,r1;x2,y2,r2\""
            ) from exc
        circles.append((x, y, r))
    return circles


def train(agent, env, dynamics_model, args):
    if hasattr(args, "load_agent"):
        agent.load_weights(args.resume)

    memory = ReplayMemory(args.replay_size, args.seed)
    total_numsteps = 0
    updates = 0

    for i_episode in range(args.max_episodes):
        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0
        done = False
        obs, _ = env.reset()

        while not done:
            if episode_steps % 10 == 0:
                prYellow("Episode {} - step {} - eps_rew {} - eps_cost {}".format( i_episode, episode_steps, episode_reward, episode_cost  )   )

            if len(memory) > args.batch_size:
                for _ in range(args.updates_per_step):
                    losses = agent.update_parameters( memory, args.batch_size, updates, dynamics_model )
                    _ = losses
                    updates += 1

            action, cbf_action = agent.select_action(obs, dynamics_model, warmup=args.start_steps > total_numsteps, safe_action=args.cbf_mode != "off",   )

            next_obs, reward, done, next_info = env.step(action)
            if "cost_exception" in next_info:
                prYellow("Cost exception occurred.")

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_cost += next_info.get("cost", 0)

            mask = 1 if episode_steps == env.max_episode_steps else float(not done)

            if args.cbf_mode == "baseline":
                memory.push(obs,  action - cbf_action, reward, next_obs, mask,)
            else:
                memory.push( obs, action,  reward,  next_obs, mask, )

            obs = next_obs
        if i_episode > 0 and i_episode % 20 == 0:
            agent.save_model(args.output)

        prGreen( "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, cost: {}".format( i_episode, total_numsteps,  episode_steps,
                round(episode_reward, 2), round(episode_cost, 2),  )       )

        if i_episode % 5 == 0 and args.eval:
            avg_reward = 0.0
            avg_cost = 0.0
            episodes = 5
            for _ in range(episodes):
                eval_obs, _ = env.reset()
                eval_done = False
                eval_rew = 0.0
                eval_cost = 0.0
                while not eval_done:
                    eval_action = agent.select_action( eval_obs,
                        dynamics_model, evaluate=True, safe_action=args.cbf_mode != "off",
                    )[0]
                    eval_obs, rew, eval_done, eval_info = env.step(eval_action)
                    eval_rew += rew
                    eval_cost += eval_info.get("cost", 0)
                avg_reward += eval_rew
                avg_cost += eval_cost
            avg_reward /= episodes
            avg_cost /= episodes
            print("----------------------------------------")
            print(
                "Test Episodes: {}, Avg. Reward: {}, Avg. Cost: {}".format(
                    episodes, round(avg_reward, 2), round(avg_cost, 2)
                )
            )
            print("----------------------------------------")


def test(agent, dynamics_model, args, visualize=True, debug=True):
    model_path = args.resume
    safe_action = args.cbf_mode != "off"
    agent.load_weights(model_path)

    def policy(observation):
        return agent.select_action(
            observation, dynamics_model, safe_action=safe_action, evaluate=True
        )[0]

    if visualize and "Unicycle" in model_path:
        from plot_utils import plot_value_function

        plot_value_function(
            build_env(
                args.env_name,
                obs_config=args.obs_config,
                rand_init=args.rand_init,
                init_mode=args.init_mode,
                init_grid_n=args.init_grid_n,
                init_x_min=args.init_x_min,
                init_x_max=args.init_x_max,
                init_y_min=args.init_y_min,
                init_y_max=args.init_y_max,
                init_theta_min=args.init_theta_min,
                init_theta_max=args.init_theta_max,
                init_v_min=args.init_v_min,
                init_v_max=args.init_v_max,
                ref_x=args.zref_x,
                ref_y=args.zref_y,
                ref_theta=args.zref_theta,
                ref_v=args.zref_v,
                r_a=args.r_a,
                r_w=args.r_w,
                custom_obstacles=getattr(args, "custom_obstacles", None),
                manual_init_state=(
                    [args.z0_x, args.z0_y, args.z0_theta, args.z0_v]
                    if None not in (args.z0_x, args.z0_y, args.z0_theta, args.z0_v)
                    else None
                ),
            ),
            agent,
            dynamics_model,
            save_path=model_path,
            safe_action=False,
        )

    episode_rewards = []
    dones = []
    for episode in range(args.validate_episodes):
        env = build_env( args.env_name, obs_config=args.obs_config,  rand_init=args.rand_init,  init_mode=args.init_mode,
            init_grid_n=args.init_grid_n, init_x_min=args.init_x_min, init_x_max=args.init_x_max, init_y_min=args.init_y_min,
            init_y_max=args.init_y_max, init_theta_min=args.init_theta_min, init_theta_max=args.init_theta_max,
            init_v_min=args.init_v_min, init_v_max=args.init_v_max, ref_x=args.zref_x, ref_y=args.zref_y,
            ref_theta=args.zref_theta,  ref_v=args.zref_v, r_a=args.r_a, r_w=args.r_w, custom_obstacles=getattr(args, "custom_obstacles", None),
            manual_init_state=( [args.z0_x, args.z0_y, args.z0_theta, args.z0_v]
                if None not in (args.z0_x, args.z0_y, args.z0_theta, args.z0_v)
                else None ), )
        
        if agent.cbf_layer:
            agent.cbf_layer.env = env

        # buffers for trajectory logging (x,y,theta)
        xs, ys, thetas = [], [], []
        state_traj = []
        action_traj = []
        observation, _ = env.reset()
        state = env.state
        if state is not None:
            xs.append(state[0]); ys.append(state[1]); thetas.append(state[2])
            state_traj.append(np.copy(state))
        episode_reward = 0.0
        done = False
        policy_timings = []
        rollout_start_t = time.time()

        while not done:
            start_t = time.time()
            action = policy(observation)
            policy_timings.append(time.time() - start_t)
            action_traj.append(np.copy(action))
            if visualize:
                env.render(mode="human")
            observation, reward, done, info = env.step(action)
            state = env.state
            if state is not None:
                xs.append(state[0]); ys.append(state[1]); thetas.append(state[2])
                state_traj.append(np.copy(state))
            episode_reward += reward
        rollout_end_t = time.time()

        episode_rewards.append(episode_reward)
        dones.append(done and env.episode_step < env.max_episode_steps)
        if debug:
            prYellow(
                "[Evaluate] #Episode{}: episode_reward:{}, mean_reward:{}, std_reward:{}, "
                "mean_completion:{}, policy_mean_wct={}".format(
                    episode, episode_reward, np.mean(episode_rewards), np.std(episode_rewards), np.mean(dones), np.mean(policy_timings),  )     )
        # convert hazards to simple dict list for plotting helpers
        obst_list = [ {"xo": h.get("location")[0], "yo": h.get("location")[1], "r": h.get("radius")}  for h in env.hazards  ]
        # save animation if requested
        if args.save_anim:
            from plot_utils import save_animation_unicycle_trajectory
            save_animation_unicycle_trajectory( xs, ys, thetas, args.initial_state_option, f"{args.anim_name}_{episode}",  obstacles=obst_list,
                robot_r=args.robot_r, margin=args.cbf_buffer, )
        if getattr(args, "save_final_shot", False):
            from plot_utils import save_unicycle_final_shot

            save_unicycle_final_shot(xs, ys, f"{args.final_shot_name}_{episode}", start_xy=(xs[0], ys[0]) if len(xs) > 0 else None,
                goal_xy=(float(env.ref_state[0]), float(env.ref_state[1])), theta_robot=thetas,  obstacles=obst_list,  robot_r=args.robot_r, margin=args.cbf_buffer,   )
        if getattr(args, "save_traj_plots", False):
            from plot_utils import save_unicycle_trajectory_diagnostics
            total_steps_sim = max(len(action_traj), 1)
            simulation_time = rollout_end_t - rollout_start_t
            time_per_step = simulation_time / total_steps_sim
            diag = save_unicycle_trajectory_diagnostics(state_traj=np.array(state_traj),  action_traj=np.array(action_traj),   dt=env.dt,  ref_state=env.ref_state,
                file_stem=f"{args.traj_plot_name}_{episode}",  output_root=args.traj_plot_dir,  plot_horizon_s=env.max_episode_steps * env.dt,pad_plot_to_horizon=True,)
            
            final_state = np.array(diag["metrics"]["final_state"])
            abs_err = float(diag["metrics"]["absolute_convergence_error"])
            avg_state_msd = float(diag["metrics"]["avg_state_msd"])
            avg_control_msd = float(diag["metrics"]["avg_control_msd"])
            
            print(f"Simulation Time {simulation_time:.2f}s,  time_per_step={time_per_step:.4f}s ")
            print(f"Average State Trajectory Mean Squared derivatives {avg_state_msd:.2f}")
            print(f"Average  Control Input Trajectory Mean Squared derivatives {avg_control_msd:.2f}") 


            print(f"Final State [{final_state[0]:.2f}, {final_state[1]:.2f}, {final_state[2]:.2f}, {final_state[3]:.2f}]" )
            print(f"Absolute Convergence Error {abs_err:.4f}")

        env.close()

    if debug:
        prYellow(
            "[Evaluate]: mean_reward:{}, std_reward:{}, mean_completion:{}".format(
                np.mean(episode_rewards), np.std(episode_rewards), np.mean(dones)
            )
        )
    return np.mean(episode_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unicycle SAC-RCBF (transfer-focused)")
    parser.add_argument("--mode", default="train", type=str, choices=["train", "test"])
    parser.add_argument("--env_name", default="Unicycle", type=str)
    parser.add_argument("--obs_config", default="default", type=str)
    parser.add_argument("--rand_init", type=parse_bool, default=False)
    parser.add_argument("--init_mode", default="manual", type=str, choices=["manual", "grid"])
    parser.add_argument("--init_grid_n", default=10, type=int)
    parser.add_argument("--init_x_min", default=-2.0, type=float)
    parser.add_argument("--init_x_max", default=2.0, type=float)
    parser.add_argument("--init_y_min", default=-2.0, type=float)
    parser.add_argument("--init_y_max", default=2.0, type=float)
    parser.add_argument("--init_theta_min", default=-np.pi, type=float)
    parser.add_argument("--init_theta_max", default=np.pi, type=float)
    parser.add_argument("--init_v_min", default=-2.0, type=float)
    parser.add_argument("--init_v_max", default=2.0, type=float)
    parser.add_argument("--zref_x", default=0.0, type=float)
    parser.add_argument("--zref_y", default=0.0, type=float)
    parser.add_argument("--zref_theta", default=0.0, type=float)
    parser.add_argument("--zref_v", default=0.0, type=float)
    parser.add_argument("--z0_x", default=None, type=float)
    parser.add_argument("--z0_y", default=None, type=float)
    parser.add_argument("--z0_theta", default=None, type=float)
    parser.add_argument("--save_anim", action="store_true", help="write trajectory animation at end of each eval episode")
    parser.add_argument("--anim_name", default="traj", type=str, help="base name for saved animation file")
    parser.add_argument("--save_final_shot", action="store_true", help="write final-shot trajectory image at end of each eval episode")
    parser.add_argument("--final_shot_name", default="final_shot", type=str, help="base name for saved final-shot image")
    parser.add_argument("--initial_state_option", default="a", type=str, choices=["a","b","c"],
                        help="option used by animation helper to choose start/goal position")
    parser.add_argument("--z0_v", default=None, type=float)
    parser.add_argument("--obs_circles", default="", type=str)
    parser.add_argument("--visualize", action="store_true", dest="visualize")
    parser.add_argument("--output", default="output", type=str)
    parser.add_argument("--policy", default="Gaussian", type=str)
    parser.add_argument("--eval", type=parse_bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--automatic_entropy_tuning", type=parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_episodes", type=int, default=3000)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--start_steps", type=int, default=5000)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=10000000)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--resume", default="default", type=str)
    parser.add_argument("--validate_episodes", default=5, type=int)
    parser.add_argument("--k_d", default=3.0, type=float)
    parser.add_argument("--gamma_b", default=20, type=float)
    parser.add_argument("--l_p", default=0.03, type=float)
    parser.add_argument("--r_a", default=1.0, type=float, help="control effort weight for acceleration: r_a * a^2")
    parser.add_argument("--r_w", default=1.0, type=float, help="control effort weight for turn-rate: r_w * omega^2")
    parser.add_argument("--robot_r", default=0.25, type=float, help="robot body radius used by CBF and animation")
    parser.add_argument("--safety_margin", default=0.05, type=float, help="extra clearance added to obstacle radius")
    parser.add_argument("--cbf_buffer", default=None, type=float, help="CBF buffer term in r_eff = r_obs + robot_r + buffer")
    parser.add_argument("--save_traj_plots", action="store_true", help="save state/control trajectory plots and metrics")
    parser.add_argument("--traj_plot_name", default="traj_diag", type=str, help="base filename for trajectory diagnostics")
    parser.add_argument("--traj_plot_dir", default="./figs/diagnostics", type=str, help="output folder for trajectory diagnostics")
    parser.add_argument("--cbf_mode", default="off", type=str, choices=["off", "baseline"])
    args = parser.parse_args()
    if args.cbf_buffer is None:
        args.cbf_buffer = args.safety_margin
    args.custom_obstacles = parse_obs_circles(args.obs_circles)

    if args.env_name != "Unicycle":
        raise Exception("Only env_name='Unicycle' is supported in this trimmed project.")

    if args.resume == "default":
        args.resume = os.getcwd() + "/output/{}-run0".format(args.env_name)
    elif args.resume.isnumeric():
        args.resume = os.getcwd() + "/output/{}-run{}".format(args.env_name, args.resume)
        args.load_agent = True

    if args.cuda:
        torch.cuda.set_device(args.device_num)

    manual_init_state = None
    if None not in (args.z0_x, args.z0_y, args.z0_theta, args.z0_v):
        manual_init_state = [args.z0_x, args.z0_y, args.z0_theta, args.z0_v]
        args.rand_init = False

    env = build_env(
        args.env_name,
        args.obs_config,
        args.rand_init,
        init_mode=args.init_mode,
        init_grid_n=args.init_grid_n,
        init_x_min=args.init_x_min,
        init_x_max=args.init_x_max,
        init_y_min=args.init_y_min,
        init_y_max=args.init_y_max,
        init_theta_min=args.init_theta_min,
        init_theta_max=args.init_theta_max,
        init_v_min=args.init_v_min,
        init_v_max=args.init_v_max,
        ref_x=args.zref_x,
        ref_y=args.zref_y,
        ref_theta=args.zref_theta,
        ref_v=args.zref_v,
        r_a=args.r_a,
        r_w=args.r_w,
        custom_obstacles=getattr(args, "custom_obstacles", None),
        manual_init_state=manual_init_state,
    )
    agent = UnicycleCBFAgent(env.observation_space.shape[0], env.action_space, env, args)
    dynamics_model = DynamicsModel(env, args)

    if args.seed > 0:
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        dynamics_model.seed(args.seed)

    if args.mode == "train":
        args.output = get_output_folder(args.output, args.env_name)
        train(agent, env, dynamics_model, args)
    else:
        test(agent, dynamics_model, args, visualize=args.visualize, debug=True)
