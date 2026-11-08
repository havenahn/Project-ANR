# Overview
This repository contains the code for the paper "An Adjoint-based Neural Regulator for Real-Time Optimal Control with State Constraints."

## Results

### **Scenario 1:  Without Obstacles navigation trajectories for CASE A, B, C for ANR, RL, and MPC**

<p align="center">
   <img alt="image 1" src="comparison_outputs/final_shot_overlays/case_a_no_obs_overlay.png" width="30%" />
  <img alt="Image 2" src="comparison_outputs/final_shot_overlays/case_b_no_obs_overlay.png"  width="30%" />
  <img alt="Image 3" src="comparison_outputs/final_shot_overlays/case_c_no_obs_overlay.png"  width="30%" />
</p>


### **Scenario 2:  With Obstacle State and Control MSD for ANR, RL, and MPC**

**Case A: Seen initial conditions and zero reference**

**CASE A: ANR**
<p align="center">
  <img alt="Image 1" src="ANR/figs/final_shot/final_shot_anr_N30_h50_a_obs2.png" width="35%" />
  <img alt="Image 2" src="ANR/figs/anr_N30_h50_a_obs2.png" width="55%" />
</p>

**CASE A: RL**
<p align="center">
  <img alt="Image 1" src="RL_SAC_HOCBF/figs/final_shot/RL_SAC_finalshot_with_2obs_caseA_0.png" width="40%" />
  <img alt="Image 2" src="RL_SAC_HOCBF/figs/diagnostics/RL_SAC_MSD_with_2obs_caseA_0.png" width="55%" />
</p>


**CASE A: MPC**
<p align="center">
  <img alt="Image 1" src="ANR/figs/final_shot/final_shot_mpc_N30_a_obs2.png" width="40%" />
  <img alt="Image 2" src="ANR/figs/mpc_N30_h50_a_obs2.png" width="55%" />
</p>


**Case B: Unseen initial conditions and zero reference**

**CASE B: ANR**
<p align="center">
  <img alt="Image 1" src="ANR/figs/final_shot/final_shot_anr_N30_h50_b_obs3.png" width="40%" />
  <img alt="Image 2" src="ANR/figs/anr_N30_h50_b_obs3.png" width="55%" />
</p>

**CASE B: MPC**
<p align="center">
  <img alt="Image 1" src="ANR/figs/final_shot/final_shot_mpc_N30_b_obs3.png" width="40%" />
  <img alt="Image 2" src="ANR/figs/mpc_N30_h50_b_obs3.png" width="55%" />
</p>



**Case C: Unseen initial conditions and nonzero reference**

**CASE C: ANR**
<p align="center">
  <img alt="Image 1" src="ANR/figs/final_shot/final_shot_anr_N30_h50_c_obs3.png" width="40%" />
  <img alt="Image 2" src="ANR/figs/anr_N30_h50_c_obs3.png" width="55%" />
</p>

**CASE C: MPC**
<p align="center">
  <img alt="Image 1" src="ANR/figs/final_shot/final_shot_mpc_N30_c_obs3.png" width="40%" />
  <img alt="Image 2" src="ANR/figs/mpc_N30_h50_c_obs3.png" width="55%" />
</p>




## Run Locally


 - _config.py_ is the file that stores the values of used parameters such as sampling time (dt), prediction horizon (N), range of Safety constraints, etc.
 - _mpc\_controller.py_ defines the Model Predictive Control (MPC) class.
 - run _train.py_ to train the Co-state Neural Network (CoNN) used in Adjoint-based Neural Regulator (NCR).
 - run _sim\_ANR.py_, _sim\_MPC.py_ to make the simulation result using ANR and MPC approach respectively.
 - run _compare\_final\_shots.py_ to visualize both ANR, RL, and MPC navigation trajectories for case A, B & C
 - run _compare\_case\_a\_random\_init\_anr\_rl.py_ to see the performance of ANR and RL evaluated over 100 initial states (**Obstacle-Free Scenarios, CASE A**).
 - run _compare\_case\_a\_random\_obstacles\_anr\_rl.py_ to see the performance of ANR and RL evaluated over 100 trials of randomized locations of two obstacles within the training domain (**Obstacles-Laden Scenarios, CASE A**).
 - run _sim\_ANR\_randomized\_obstacles.py_ to see the performance of ANR and RL evaluated over 100 trials of randomized locations of three obstacles outside the training domain (**Obstacles-Laden Scenarios, CASE B**).



# SQP-RTI RESULTS

### **Scenario 1:  Without Obstacle State and Control MSD**

**Case A: Seen initial conditions and zero reference**
<p align="center">
   <img alt="Image 1" src="ANR/figs/final_shot/final_shot_mpc_acados_rti_N30_a_no_obs.png" width="38%" />
   <img alt="Image 2" src="ANR/figs/mpc_N30_h50_acados_rti_a_no_obs.png" width="55%" />
</p>

**Case B: Unseen initial conditions and zero reference**
<p align="center">
   <img alt="Image 1" src="ANR/figs/final_shot/final_shot_mpc_acados_rti_N30_b_no_obs.png" width="38%" />
   <img alt="Image 2" src="ANR/figs/mpc_N30_h50_acados_rti_b_no_obs.png" width="55%" />
</p>


**Case C: Unseen initial conditions and nonzero reference**
<p align="center">
   <img alt="Image 1" src="ANR/figs/final_shot/final_shot_mpc_acados_rti_N30_c_no_obs.png" width="40%" />
  <img alt="Image 2" src="ANR/figs/mpc_N30_h50_acados_rti_c_no_obs.png" width="55%" />
</p>


### **Scenario 2:  Without Obstacle State and Control MSD**

**Case A: Seen initial conditions and zero reference**
<p align="center">
   <img alt="Image 1" src="ANR/figs/final_shot/final_shot_mpc_acados_rti_N30_a_obs2.png" width="38%" />
   <img alt="Image 2" src="ANR/figs/mpc_N30_h50_acados_rti_a_obs2.png" width="55%" />
</p>

**Case B: Unseen initial conditions and zero reference**
<p align="center">
   <img alt="Image 1" src="ANR/figs/final_shot/final_shot_mpc_acados_rti_N30_b_obs3.png" width="38%" />
   <img alt="Image 2" src="ANR/figs/mpc_N30_h50_acados_rti_b_obs3.png" width="55%" />
</p>


**Case C: Unseen initial conditions and nonzero reference**
<p align="center">
   <img alt="Image 1" src="ANR/figs/final_shot/final_shot_mpc_acados_rti_N30_c_obs3.png" width="40%" />
  <img alt="Image 2" src="ANR/figs/mpc_N30_h50_acados_rti_c_obs3.png" width="55%" />
</p>




 # Extra help file for RL


## The objectives are

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

## Deployment
Test zero-shot transfer with random obstacles and HOCBF safety enabled:

You can save a matplotlib animation of the unicycle rolling along its path by adding the `--save_anim` flag (and optionally `--anim_name`/`--initial_state_option`). The file will be written to `./animation/unicycle/<anim_name>_<episode>.mp4`.



## Random initial states (this is to randomize different initial states within the training domain to see if the robot can get to the reference given any initial state)
``` bash
cd .\RL_SAC_HOCBF
python RL_case_a_init_state_randomized.py --resume 1 --num_trials 100 --success_threshold 0.6 --d_min 0.5 --cbf_mode off --seed 0 --init_v_min 0  --init_v_max 0 --max_failed_print 1

```


## Randomized obstacles(Given an initial state in the training domain, this is to randomize obstacles between the initial state and the reference )
```bash
cd .\RL_SAC_HOCBF
python RL_obstacles_randomized.py --resume 1 --case_option a --num_trials 100 --success_threshold 0.6 --cbf_mode baseline --seed 0 --robot_r 0.25 --cbf_buffer 0.05 --obstacle_min_sep 0.5

```


**NOTE:** use (z0_x, z0_y, z0_theta, z0_v) to define the initial states you want and (zref_x, zref_y, zref_theta, zref_v) to define the reference

## Without obstacle test - case a  
```bash

cd .\RL_SAC_HOCBF
python main.py --mode test --env_name Unicycle --resume 1 --cbf_mode off --rand_init False --obs_config none --z0_x -1.9 --z0_y 2 --z0_theta -1.79 --z0_v 0 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --validate_episodes 1 --save_anim --anim_name RL_SAC_no_obs_caseA --save_final_shot --final_shot_name RL_SAC_finalshot_no_obs_casA --save_traj_plots --traj_plot_name RL_SAC_MSD_no_obs_caseA --traj_plot_dir ./figs/diagnostics --initial_state_option a --seed 12345

```


## With obstacle test - case a   

```bash
cd .\RL_SAC_HOCBF
python .\main.py --mode test --env_name Unicycle --resume 1 --cbf_mode baseline --rand_init False --obs_circles="-0.5,1.1,0.1; -1.5,1.2,0.1" --z0_x -1.9 --z0_y 2 --z0_theta -1.79 --z0_v 0 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --validate_episodes 1 --save_anim --anim_name RL_SAC_with_2obs_caseA --save_final_shot --final_shot_name RL_SAC_finalshot_with_2obs_caseA --save_traj_plots --traj_plot_name RL_SAC_MSD_with_2obs_caseA --traj_plot_dir .\figs\diagnostics --initial_state_option a --seed 12345

```


## Without obstacle test - case b

```bash
cd .\RL_SAC_HOCBF
python main.py --mode test --env_name Unicycle --resume 1 --cbf_mode off --rand_init False --obs_config none --z0_x -5.24 --z0_y 4.11 --z0_theta 2.72 --z0_v 0 --zref_x 0 --zref_y 0 --zref_theta 0 --zref_v 0 --robot_r 0.25 --cbf_buffer 0.05 --validate_episodes 1 --save_anim --anim_name RL_SAC_no_obs_caseb --save_final_shot --final_shot_name RL_SAC_finalshot_no_obs_caseb --save_traj_plots --traj_plot_name RL_SAC_MSD_no_obs_caseb --traj_plot_dir ./figs/diagnostics --initial_state_option b --seed 12345

```



