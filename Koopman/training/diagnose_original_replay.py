#!/usr/bin/env python3
"""Original Koopman one-step replay diagnostic.

Expected layout:
Project_ANR/Koopman/data/Fx_5_50001_0_file0.npy
Project_ANR/Koopman/models/model.pt
Project_ANR/Koopman/training/networkarch.py
Place this file in Koopman/training/ and run:
    python diagnose_original_replay.py
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path,
        default=KOOPMAN_DIR / "data" / "Fx_5_50001_0_file0.npy"
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=KOOPMAN_DIR / "models" / "model42.pt"
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-trajectories", type=int, default=0,
                        help="0 evaluates all trajectories")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data: {args.data.resolve()}")
    print(f"Checkpoint: {args.checkpoint.resolve()}")

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Default is Koopman/models/model.pt; use --checkpoint if needed."
        )

    raw = np.load(args.data)
    print(f"Raw data shape: {raw.shape}")
    if raw.ndim != 3 or raw.shape[1] < 2 or raw.shape[2] < 8:
        raise ValueError(
            "Expected (trajectories, at least 2 time steps, at least 8 columns); "
            f"got {raw.shape}"
        )

    # Same column slicing as training.py:
    # data[:, :3, 2:] -> first 4 columns are state, last 2 are control.
    x0 = raw[:, 0, 2:6].astype(np.float32, copy=False)
    u0 = raw[:, 0, 6:8].astype(np.float32, copy=False)
    x1_true = raw[:, 1, 2:6].astype(np.float32, copy=False)

    if args.max_trajectories:
        n = min(args.max_trajectories, len(x0))
        x0, u0, x1_true = x0[:n], u0[:n], x1_true[:n]

    if not all(np.isfinite(a).all() for a in (x0, u0, x1_true)):
        raise ValueError("Selected dataset columns contain NaN or Inf.")

    print(f"Trajectories evaluated: {len(x0)}")
    print("Input ranges:")
    for i, name in enumerate(STATE_NAMES):
        print(f"  {name:8s}: [{x0[:, i].min(): .7g}, {x0[:, i].max(): .7g}]")
    for i, name in enumerate(("u1", "u2")):
        print(f"  {name:8s}: [{u0[:, i].min(): .7g}, {u0[:, i].max(): .7g}]")

    model = NeuralNetwork(N_x=4, N_h=32, N_e=100).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)

    # Original training code saves {'model_dict': model.state_dict()}.
    state_dict = checkpoint["model_dict"] if (
        isinstance(checkpoint, dict) and "model_dict" in checkpoint
    ) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    pred_batches = []
    with torch.inference_mode():
        for start in range(0, len(x0), args.batch_size):
            end = min(start + args.batch_size, len(x0))
            xb = torch.from_numpy(x0[start:end]).to(device)
            ub = torch.from_numpy(u0[start:end]).to(device)
            x1_pred, _ = model(xb, ub)
            pred_batches.append(x1_pred.cpu().numpy())

    x1_pred = np.concatenate(pred_batches, axis=0)
    error = x1_pred - x1_true
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    max_abs = np.max(np.abs(error), axis=0)

    print("\nOne-step replay: x0, u0 -> predicted x1")
    print(f"{'Component':<12} {'MAE':>14} {'RMSE':>14} {'Max abs error':>16}")
    print("-" * 60)
    for i, name in enumerate(STATE_NAMES):
        print(f"{name:<12} {mae[i]:14.7g} {rmse[i]:14.7g} {max_abs[i]:16.7g}")
    print("-" * 60)
    print(
        f"{'ALL values':<12} "
        f"{np.mean(np.abs(error)):14.7g} "
        f"{np.sqrt(np.mean(error ** 2)):14.7g} "
        f"{np.max(np.abs(error)):16.7g}"
    )

    print("\nTarget vs prediction ranges:")
    for i, name in enumerate(STATE_NAMES):
        print(
            f"{name:8}: target [{x1_true[:, i].min(): .7g}, "
            f"{x1_true[:, i].max(): .7g}] | pred "
            f"[{x1_pred[:, i].min(): .7g}, {x1_pred[:, i].max(): .7g}]"
        )

    print("\nFinished. This is one-step replay only; no recursive rollout was run.")


if __name__ == "__main__":
    main()
