
import argparse
import numpy as np
import torch
from pathlib import Path
from networkarch import NeuralNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="../data/Fx_31_2000_123_zero_control.npy"
    )
    parser.add_argument(
        "--checkpoint",
        default="../models/model.pt"
    )
    parser.add_argument(
        "--output",
        default="zero_teacher_forced_results.npz"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    raw = np.load(args.data)
    x_true = raw[:, :, 2:6].astype(np.float32)
    u_true = raw[:, :, 6:8].astype(np.float32)

    n_traj, n_steps, _ = x_true.shape
    print("Data shape:", raw.shape)
    print("Max |control|:", np.abs(u_true).max())

    # Load model
    model = NeuralNetwork(N_x=4, N_h=32, N_e=100).to(device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_dict"])
    model.eval()

    # Teacher-forced one-step prediction
    predictions = []
    targets = []

    with torch.no_grad():
        for k in range(n_steps - 1):
            xk = torch.tensor(x_true[:, k, :], device=device)

            # Re-encode the TRUE state at every timestep
            eta = model.enc(xk)
            xi = torch.cat([xk, eta], dim=1)

            # Zero-control dynamics: xi_next = A(xi)
            xi_next = model.A(xi)
            x_pred = xi_next[:, :4]

            predictions.append(x_pred.cpu().numpy())
            targets.append(x_true[:, k + 1, :])

    predictions = np.stack(predictions, axis=1)
    targets = np.stack(targets, axis=1)

    # Overall metrics
    error = predictions - targets
    component_names = ["pbody1", "pbody2", "x_dot", "y_dot"]

    print("\n=== Overall teacher-forced error ===")
    for i, name in enumerate(component_names):
        mae = np.mean(np.abs(error[:, :, i]))
        rmse = np.sqrt(np.mean(error[:, :, i] ** 2))
        max_error = np.max(np.abs(error[:, :, i]))
        print(
            f"{name:8s} | MAE={mae:.6f} "
            f"RMSE={rmse:.6f} MAX={max_error:.6f}"
        )

    # Per-step metrics
    print("\n=== Per-step RMSE ===")
    print("step | pbody1 | pbody2 | x_dot | y_dot")

    step_rmse = np.sqrt(np.mean(error ** 2, axis=0))

    for k in range(n_steps - 1):
        values = step_rmse[k]
        print(
            f"{k+1:4d} | "
            + " | ".join(f"{v:.6f}" for v in values)
        )

    # Save results
    np.savez(
        args.output,
        predictions=predictions,
        targets=targets,
        errors=error,
        step_rmse=step_rmse
    )

    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
