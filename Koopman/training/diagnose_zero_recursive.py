#!/usr/bin/env python3
"""Zero-control recursive Koopman rollout diagnostic.

Expected layout:
Project_ANR/
└── Koopman/
    ├── data/Fx_31_2000_123_zero_control.npy
    ├── models/model.pt
    └── training/
        ├── networkarch.py
        └── diagnose_zero_recursive.py  <- this file

Run from Koopman/training:
    python diagnose_zero_recursive.py

Optional:
    python diagnose_zero_recursive.py --checkpoint ../models/model42.pt
    python diagnose_zero_recursive.py --data ../data/Fx_31_2000_123_zero_control.npy
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
KOOPMAN_DIR = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from networkarch import NeuralNetwork

STATE_NAMES = ["pbody1", "pbody2", "x_dot", "y_dot"]


def load_model(checkpoint_path: Path, device: torch.device) -> NeuralNetwork:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = NeuralNetwork(N_x=4, N_h=32, N_e=100).to(device)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = (
        checkpoint["model_dict"]
        if isinstance(checkpoint, dict) and "model_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Recursive zero-control rollout using xi[k+1] = A xi[k]."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=KOOPMAN_DIR / "data" / "Fx_31_2000_123_zero_control.npy",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=KOOPMAN_DIR / "models" / "model.pt",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="0 evaluates all trajectories.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=1,
        help="Print metrics every N transitions.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data: {args.data.resolve()}")
    print(f"Checkpoint: {args.checkpoint.resolve()}")

    if not args.data.is_file():
        raise FileNotFoundError(
            f"Zero-control dataset not found: {args.data}\n"
            "Expected: Koopman/data/Fx_31_2000_123_zero_control.npy"
        )

    raw = np.load(args.data)
    print(f"Raw data shape: {raw.shape}")

    if raw.ndim != 3 or raw.shape[1] < 2 or raw.shape[2] < 8:
        raise ValueError(
            "Expected shape (trajectories, time_steps>=2, columns>=8); "
            f"received {raw.shape}"
        )

    # Same raw-column convention used by the original training script:
    # columns 2:6 are physical state; columns 6:8 are controls.
    x_true = raw[:, :, 2:6].astype(np.float32, copy=False)
    u_true = raw[:, :, 6:8].astype(np.float32, copy=False)

    n_traj, n_steps, _ = x_true.shape
    if args.max_trajectories > 0:
        n_traj = min(n_traj, args.max_trajectories)
        x_true = x_true[:n_traj]
        u_true = u_true[:n_traj]

    if not np.isfinite(x_true).all() or not np.isfinite(u_true).all():
        raise ValueError("Dataset contains NaN or Inf.")

    max_control = float(np.max(np.abs(u_true)))
    print(f"Trajectories: {n_traj}; timesteps: {n_steps}")
    print(f"Maximum absolute recorded control: {max_control:.8g}")
    if max_control > 1e-8:
        print("WARNING: this file is not strictly zero-control; continuing anyway.")

    model = load_model(args.checkpoint, device)

    # Recursive rollout: encode only the initial true state.
    # Thereafter apply A repeatedly. Since u=0, B(xi)u = 0.
    predictions = np.full_like(x_true, np.nan, dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, n_traj, args.batch_size):
            end = min(start + args.batch_size, n_traj)
            x0 = torch.from_numpy(x_true[start:end, 0, :]).to(device)
            eta0 = model.enc(x0)
            xi = torch.cat((x0, eta0), dim=1)

            predictions[start:end, 0, :] = x0.cpu().numpy()

            for k in range(1, n_steps):
                xi = model.A(xi)
                x_pred = xi[:, :4]
                predictions[start:end, k, :] = x_pred.cpu().numpy()

    print("\nRecursive rollout errors against recorded zero-control trajectories:")
    print(f"{'step':>5} {'component':<10} {'MAE':>13} {'RMSE':>13} {'Max abs':>13}")
    print("-" * 60)

    errors = predictions - x_true
    for k in range(1, n_steps):
        if k % args.report_every != 0 and k != n_steps - 1:
            continue
        for j, name in enumerate(STATE_NAMES):
            e = errors[:, k, j]
            mae = np.mean(np.abs(e))
            rmse = np.sqrt(np.mean(e ** 2))
            max_abs = np.max(np.abs(e))
            print(f"{k:5d} {name:<10} {mae:13.6g} {rmse:13.6g} {max_abs:13.6g}")

    print("\nPredicted physical-state magnitude by step (across all trajectories):")
    print(f"{'step':>5} {'max |state|':>16} {'mean |state|':>16} {'finite':>10}")
    print("-" * 52)
    for k in range(n_steps):
        step_pred = predictions[:, k, :]
        finite = np.isfinite(step_pred)
        finite_count = int(finite.all(axis=1).sum())
        if finite.any():
            max_abs = np.nanmax(np.abs(step_pred))
            mean_abs = np.nanmean(np.abs(step_pred))
        else:
            max_abs = float("nan")
            mean_abs = float("nan")
        print(f"{k:5d} {max_abs:16.7g} {mean_abs:16.7g} {finite_count:10d}")

    # Save numerical outputs beside the script for later plotting/comparison.
    output_path = SCRIPT_DIR / "zero_recursive_rollout_results.npz"
    np.savez_compressed(
        output_path,
        x_true=x_true,
        x_pred=predictions,
        errors=errors,
    )
    print(f"\nSaved predictions and errors to: {output_path}")
    print("Done. This is recursive zero-control rollout; it does not re-encode true states.")


if __name__ == "__main__":
    main()
