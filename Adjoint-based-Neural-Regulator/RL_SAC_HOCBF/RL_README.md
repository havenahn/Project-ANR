# RL_help file


# The objectives are

1. Train a Unicycle RL policy with no obstacles.
2. Deploy/test that policy with obstacles and a CBF safety filter.

## Supported scope

- Environment: `Unicycle` only
- Modes: `train`, `test`
- CBF modes: `off` (train), `baseline` (deploy/test)

## Main workflow

Notes:
- If `--resume` is a number (for example `1`), it resolves to `output/Unicycle-run1`.
- Saved model files are under `output/Unicycle-run*/`.
- Effective HOCBF/animation obstacle boundary is `r_eff = obstacle_radius + robot_r + safety_margin`.

## To Run Locally

## For training 
```bash
cd .\RL_SAC_HOCBF
python main.py --mode train --env_name Unicycle --cbf_mode off --obs_config none --rand_init True --init_mode grid --init_grid_n 10 --init_x_min -2 --init_x_max 2 --init_y_min -2 --init_y_max 2 --init_theta_min -2 --init_theta_max 2 --init_v_min -2 --init_v_max 2 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --seed 12345 

```

# Deployment
Test zero-shot transfer with random obstacles and HOCBF safety enabled:

You can save a matplotlib animation of the unicycle rolling along its path by adding the `--save_anim` flag (and optionally `--anim_name`/`--initial_state_option`). The file will be written to `./animation/unicycle/<anim_name>_<episode>.mp4`.



# random intial states (this is to randomized different initial states within the training domain to see if the robot can get to the reference given any initial state)
``` bash
cd .\RL_SAC_HOCBF
python RL_case_a_init_state_randomized.py --resume 1 --num_trials 100 --success_threshold 0.6 --d_min 0.5 --cbf_mode off --seed 0 --init_v_min 0  --init_v_max 0 --max_failed_print 1

```


# randomized obstacles(Given an initial state in the training domain, this is to randomized obstacles between the initial state and the reference )
```bash
cd .\RL_SAC_HOCBF
python RL_obstacles_randomized.py --resume 1 --case_option a --num_trials 100 --success_threshold 0.6 --cbf_mode baseline --seed 0 --robot_r 0.25 --cbf_buffer 0.05 --obstacle_min_sep 0.5

```


#NOTE: use (z0_x, z0_y, z0_theta, z0_v) to define the initial states you want and (zref_x, zref_y, zref_theta, zref_v) to define the reference

# Without obstacle test - case a  
```bash

cd .\RL_SAC_HOCBF
python main.py --mode test --env_name Unicycle --resume 1 --cbf_mode off --rand_init False --obs_config none --z0_x -1.9 --z0_y 2 --z0_theta -1.79 --z0_v 0 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --validate_episodes 1 --save_anim --anim_name RL_SAC_no_obs_caseA --save_final_shot --final_shot_name RL_SAC_finalshot_no_obs_casA --save_traj_plots --traj_plot_name RL_SAC_MSD_no_obs_caseA --traj_plot_dir ./figs/diagnostics --initial_state_option a --seed 12345

```


# With obstacle test - case a   

```bash
cd .\RL_SAC_HOCBF
python .\main.py --mode test --env_name Unicycle --resume 1 --cbf_mode baseline --rand_init False --obs_circles="-0.5,1.1,0.1; -1.5,1.2,0.1" --z0_x -1.9 --z0_y 2 --z0_theta -1.79 --z0_v 0 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --validate_episodes 1 --save_anim --anim_name RL_SAC_with_2obs_caseA --save_final_shot --final_shot_name RL_SAC_finalshot_with_2obs_caseA --save_traj_plots --traj_plot_name RL_SAC_MSD_with_2obs_caseA --traj_plot_dir .\figs\diagnostics --initial_state_option a --seed 12345

```


# Without obstacle test - case b

```bash
cd .\RL_SAC_HOCBF
python main.py --mode test --env_name Unicycle --resume 1 --cbf_mode off --rand_init False --obs_config none --z0_x -5.24 --z0_y 4.11 --z0_theta 2.72 --z0_v 0 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --robot_r 0.25 --cbf_buffer 0.05 --validate_episodes 1 --save_anim --anim_name RL_SAC_no_obs_caseb --save_final_shot --final_shot_name RL_SAC_finalshot_no_obs_caseb --save_traj_plots --traj_plot_name RL_SAC_MSD_no_obs_caseb --traj_plot_dir ./figs/diagnostics --initial_state_option b --seed 12345

```
