#!/usr/bin/env python3
"""Compare zero-control teacher-forced prediction errors inside/outside
training-data state min/max ranges.

Run from Koopman/training:
    python compare_in_out_range_errors.py

Expected files:
    ../data/Fx_5_50001_0_file0.npy
    ../data/Fx_31_2000_123_zero_control.npy
    zero_teacher_forced_results.npz
"""

import argparse
import csv
from pathlib import Path
import numpy as np

NAMES = ["pbody1", "pbody2", "x_dot", "y_dot"]


def metrics(err):
    """err shape: (n_samples, 4)"""
    if err.shape[0] == 0:
        return None
    return {
        "n": int(err.shape[0]),
        "mae": np.mean(np.abs(err), axis=0),
        "rmse": np.sqrt(np.mean(err ** 2, axis=0)),
        "max_abs": np.max(np.abs(err), axis=0),
    }


def print_group(label, result):
    if result is None:
        print(f"  {label}: N=0 (no samples)")
        return
    print(f"  {label}: N={result['n']}")
    for j, name in enumerate(NAMES):
        print(
            f"    {name:8s} RMSE={result['rmse'][j]:.6f} "
            f"MAE={result['mae'][j]:.6f} "
            f"MAX={result['max_abs'][j]:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="../data/Fx_5_50001_0_file0.npy")
    parser.add_argument("--zero-data", default="../data/Fx_31_2000_123_zero_control.npy")
    parser.add_argument("--results", default="../data/zero_teacher_forced_results.npz")
    parser.add_argument("--csv", default="../data/in_out_range_error_comparison.csv")
    args = parser.parse_args()

    for filename in (args.train_data, args.zero_data, args.results):
        if not Path(filename).is_file():
            raise FileNotFoundError(
                f"Could not find: {filename}\n"
                "Run this script from Koopman/training, or pass the correct paths."
            )

    train_raw = np.load(args.train_data)
    zero_raw = np.load(args.zero_data)
    saved = np.load(args.results)

    # Match the original training.py slicing: first 3 time samples, state columns 2:6.
    train_x = train_raw[:, :3, 2:6].reshape(-1, 4)
    train_min = train_x.min(axis=0)
    train_max = train_x.max(axis=0)

    # A teacher-forced prediction at transition k -> k+1 uses the TRUE x_k.
    true_input_states = zero_raw[:, :-1, 2:6].astype(np.float64)
    predictions = saved["predictions"].astype(np.float64)
    targets = saved["targets"].astype(np.float64)
    errors = saved["errors"].astype(np.float64)

    expected_shape = true_input_states.shape
    if predictions.shape != expected_shape or targets.shape != expected_shape or errors.shape != expected_shape:
        raise ValueError(
            "Shape mismatch: expected predictions/targets/errors to have "
            f"shape {expected_shape}, got {predictions.shape}, {targets.shape}, {errors.shape}."
        )

    # Check that saved errors really correspond to prediction - target.
    discrepancy = np.max(np.abs(errors - (predictions - targets)))
    if discrepancy > 1e-5:
        print(f"WARNING: saved errors differ from predictions-targets (max diff={discrepancy:.3e}); recomputing.")
        errors = predictions - targets

    # State is in-range only if ALL FOUR coordinates fall inside their
    # corresponding training min/max intervals.
    inside_each_coordinate = (
        (true_input_states >= train_min[None, None, :])
        & (true_input_states <= train_max[None, None, :])
    )
    in_range = np.all(inside_each_coordinate, axis=2)
    out_of_range = ~in_range

    n_traj, n_transitions, _ = errors.shape
    print("=== Setup ===")
    print(f"Training data: {train_raw.shape}; using timesteps 0-2")
    print(f"Zero-control data: {zero_raw.shape}")
    print(f"Teacher-forced predictions: {predictions.shape}")
    print("Training min/max bounds:")
    for j, name in enumerate(NAMES):
        print(f"  {name:8s}: [{train_min[j]:.6f}, {train_max[j]:.6f}]")

    total_in = int(in_range.sum())
    total_out = int(out_of_range.sum())
    total = total_in + total_out
    print("\n=== Pooled results across all transitions ===")
    print(f"All four state variables in range: {total_in}/{total} ({100*total_in/total:.2f}%)")
    print(f"At least one state variable out of range: {total_out}/{total} ({100*total_out/total:.2f}%)")
    print_group("IN-RANGE", metrics(errors[in_range]))
    print_group("OUT-OF-RANGE", metrics(errors[out_of_range]))

    print("\n=== Per-transition comparison ===")
    print("Transition is k -> k+1; classification uses true input state x_k.")

    csv_rows = []
    for k in range(n_transitions):
        mask_in = in_range[:, k]
        mask_out = out_of_range[:, k]
        err_k = errors[:, k, :]
        result_in = metrics(err_k[mask_in])
        result_out = metrics(err_k[mask_out])

        print(
            f"\nTransition {k+1:2d} (input timestep {k:2d}): "
            f"in={int(mask_in.sum())}, out={int(mask_out.sum())}"
        )
        print_group("IN-RANGE", result_in)
        print_group("OUT-OF-RANGE", result_out)

        for group_name, result in (("in_range", result_in), ("out_of_range", result_out)):
            if result is None:
                continue
            for j, name in enumerate(NAMES):
                csv_rows.append({
                    "transition": k + 1,
                    "input_timestep": k,
                    "group": group_name,
                    "n_samples": result["n"],
                    "variable": name,
                    "rmse": float(result["rmse"][j]),
                    "mae": float(result["mae"][j]),
                    "max_abs_error": float(result["max_abs"][j]),
                })

    output_path = Path(args.csv)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "transition", "input_timestep", "group", "n_samples",
                "variable", "rmse", "mae", "max_abs_error"
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved per-transition comparison to: {output_path}")
    print("Note: min/max membership is a simple range check, not a full joint-distribution test.")


if __name__ == "__main__":
    main()
