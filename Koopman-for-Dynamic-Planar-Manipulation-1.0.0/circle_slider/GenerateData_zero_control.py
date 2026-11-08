#!/usr/bin/env python3
"""Generate long zero-input trajectories with the same planar-pusher simulator.

Run from the Koopman model directory, i.e. the directory containing
planar_pusher.py, GenerateData_subset.py, and training(5).py.
"""
from pathlib import Path
import numpy as np

from planar_pusher import Pusher
from GenerateData_subset import generate_data


if __name__ == "__main__":
    # Match GenerateData_main.py physical parameters and state domains.
    traj_len = 31                 # x_0 through x_30: 30 model transitions
    desired_sample_dt = 0.125     # matches the spacing used by the original 5-point data
    # generate_data currently builds linspace(0, traj_len * dt, traj_len).
    # Choose dt so the resulting adjacent sample spacing is exactly 0.125 s.
    generator_dt = desired_sample_dt * (traj_len - 1) / traj_len

    num_trajectories = 2000
    seed = 123

    block_diameter = 0.16
    block_mass = 0.827
    table_mu = 0.35
    pusher_mu = 0.3
    pusher_k = 1000
    max_energy = None
    num_states = 12

    x_domain = np.array([-block_diameter / 20, block_diameter / 20])
    y_domain = np.array([-block_diameter / 20, block_diameter / 20])
    pbody1_domain = np.array([-0.02, 0.02])
    pbody2_domain = np.array([-np.pi / 2, np.pi / 2])
    x_dot_domain = np.array([-0.3, 0.3])
    y_dot_domain = np.array([-0.3, 0.3])
    domains = [
        x_domain,
        y_domain,
        pbody1_domain,
        pbody2_domain,
        x_dot_domain,
        y_dot_domain,
    ]

    init_state = np.array([0, 0, block_diameter / 2, 0, 0, 0])
    pusher = Pusher(
        block_diameter, block_mass, table_mu, pusher_mu, pusher_k, init_state
    )
    conduit_traj = np.array([pusher.state])

    generated = generate_data(
        conduit_traj=conduit_traj,
        max_energy=max_energy,
        traj_len=traj_len,
        pusher=pusher,
        save_path=".",
        num_states=num_states,
        dt=generator_dt,
        domains=domains,
        SAVE=False,
        VERBOSE=False,
        duration=1.0,
        seed=seed,
        file_array_size=num_trajectories - 1,
        file_count=0,
        control_mode="zero",
    )

    generated = np.asarray(generated, dtype=np.float64)
    out_dir = Path("../data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"Fx_{traj_len}_{num_trajectories}_{seed}_zero_control.npy"
    np.save(out_path, generated)

    print("Saved:", out_path.resolve())
    print("Shape:", generated.shape)
    print("Control max abs:", np.abs(generated[:, :, -2:]).max())
