
import torch
from pathlib import Path
from koopman_dynamics import load_koopman_model

PROJECT_ROOT = Path.home() / "Project_ANR"

KOOPMAN_REPO = (
    PROJECT_ROOT
    / "Koopman-for-Dynamic-Planar-Manipulation-1.0.0"
    / "circle_slider"
    / "bilinear"
    / "nn_multistep_loss_1_bilinear_mu035_copy"
)

CHECKPOINT = KOOPMAN_REPO / "model.pt"

model, device = load_koopman_model(CHECKPOINT)

print("Model loaded successfully!")
print("Device:", device)
print("Checkpoint:", CHECKPOINT)

# Dummy physical state and control input
x = torch.randn(1, 4, device=device)
u = torch.randn(1, 2, device=device)

with torch.no_grad():
    x_next, eta_next = model(x, u)

print("Input state shape:", x.shape)
print("Control shape:", u.shape)
print("Predicted state shape:", x_next.shape)
print("Predicted latent shape:", eta_next.shape)
print("Predicted state:", x_next.cpu().numpy())

assert x_next.shape == (1, 4)
assert eta_next.shape == (1, 100)

print("All basic checks passed!")