from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path.home() / "Project_ANR"

KOOPMAN_REPO = (
    PROJECT_ROOT
    / "Koopman-for-Dynamic-Planar-Manipulation-1.0.0"
    / "circle_slider/bilinear"
    /"nn_multistep_loss_1_bilinear_mu035_copy/"
)

CHECKPOINT = (
    KOOPMAN_REPO
    / "model.pt"
)

sys.path.insert(0, str(KOOPMAN_REPO))
from helpers.networkarch import NeuralNetwork


def load_koopman_model(checkpoint_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Use the same architecture as training
    model = NeuralNetwork(
        N_x=4,
        N_h=32,
        N_e=100,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # L3.learn() saves a dictionary containing model_dict
    state_dict = checkpoint.get("model_dict", checkpoint)
    model.load_state_dict(state_dict)

    model.eval()
    return model, device