#!/usr/bin/env python3
"""Fine-tune the pretrained bilinear Koopman model with long zero-input data.

Run from the same model directory where the original training(5).py is run.
Expected paths: ./model.pt and ../data/Fx_5_50001_0_file0.npy.
"""
from pathlib import Path
import numpy as np

from helpers.learn_modules_zeroaug import L3ZeroControlFineTune


if __name__ == "__main__":
    data_dir = Path("../data")
    controlled_path = data_dir / "Fx_5_50001_0_file0.npy"
    zero_path = data_dir / "Fx_31_2000_123_zero_control.npy"

    if not controlled_path.exists():
        raise FileNotFoundError(f"Missing controlled data: {controlled_path.resolve()}")
    if not zero_path.exists():
        raise FileNotFoundError(
            f"Missing zero-control data: {zero_path.resolve()}\n"
            "Run GenerateData_zero_control.py first."
        )

    raw_controlled = np.load(controlled_path)
    raw_zero = np.load(zero_path)

    # Preserve the original training preprocessing for the controlled data.
    controlled_data = raw_controlled[:, :3, 2:].astype(np.float32)
    # Use the full 31-point zero-input trajectories to supervise 30 transitions.
    zero_control_data = raw_zero[:, :, 2:].astype(np.float32)

    trainer = L3ZeroControlFineTune(
        initial_checkpoint="model.pt",
        output_checkpoint="model_zeroaug.pt",
        n_x=4,
        n_h=32,
        n_e=100,
        epochs=50,
        controlled_batch_size=8192,
        zero_batch_size=128,
        learning_rate=1e-5,
        zero_loss_weight=2.0,
        gradient_clip=1.0,
        seed=42,
    )
    trainer.fit(controlled_data, zero_control_data)
