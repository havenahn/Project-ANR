#!/usr/bin/env python3


"""Fine-tune the pretrained Koopman A matrix using the actual project layout.
directory kind of messy rn

Koopman-for-Dynamic-Planar-Manipulation-1.0.0/
├── data/Fx_31_2000_123_zero_control.npy
└── circle_slider/
    ├── training_zeroaug.py
    └── bilinear/
        └── data/Fx_5_50001_0_file0.npy

The checkpoint is searched in the known Koopman model folders
"""
from pathlib import Path
import numpy as np


from bilinear.nn_multistep_loss_1_bilinear_mu035_copy.helpers.learn_modules_zeroaug import L3ZeroControlFineTune

BASE_DIR = Path(__file__).resolve().parent       # .../circle_slider
PROJECT_ROOT = BASE_DIR.parent                   # .../Koopman-for-Dynamic-Planar-Manipulation-1.0.0


def first_existing_file(candidates, description):
    checked = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        checked.append(candidate)
        if candidate.is_file():
            return candidate
    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        f"Could not find {description}. Checked these paths:\n{checked_text}"
    )

if __name__ == "__main__":
     # The controlled training data is inside circle_slider/bilinear/data.
    controlled_path = first_existing_file(
        [
            BASE_DIR / "bilinear" / "data" / "Fx_5_50001_0_file0.npy",
            BASE_DIR / "bilinear" / "nn_multistep_loss_1_bilinear_mu035_copy" / "data" / "Fx_5_50001_0_file0.npy",
        ],
        "original controlled training data Fx_5_50001_0_file0.npy",
    )

    # The zero-control generator saved this file in the repository-level data folder,
    # which is a sibling of circle_slider (as shown in the first screenshot).
    zero_path = first_existing_file(
        [
            PROJECT_ROOT / "data" / "Fx_31_2000_123_zero_control.npy",
            BASE_DIR / "data" / "Fx_31_2000_123_zero_control.npy",
        ],
        "zero-control data Fx_31_2000_123_zero_control.npy",
    )
    # Prefer the checkpoint location already used by the ANR script.
    checkpoint_path = first_existing_file(
        [
            BASE_DIR / "bilinear" / "nn_multistep_loss_1_bilinear_mu035_copy" / "model.pt",
            BASE_DIR / "bilinear" / "model.pt",
            BASE_DIR / "model.pt",
            Path.cwd() / "model.pt",
        ],
        "pretrained checkpoint model.pt",
    )
    output_path = checkpoint_path.with_name("model_zeroaug.pt")

    print("Controlled data:", controlled_path)
    print("Zero-control data:", zero_path)
    print("Initial checkpoint:", checkpoint_path)
    print("Output checkpoint:", output_path)

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
