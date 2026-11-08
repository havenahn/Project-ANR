import torch

#load checkpoint
checkpoint_path = (
    "/Users/havenahn/Project_ANR/"
    "Adjoint-based-Neural-Regulator/ANR/model/"
    "anr_N30_h50_seed_0_e50.pth"
)

#load the costate neural network architecture as is
from train_anr_zero_goal import CoNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

anr_model = CoNN(
    prediction_horizon=30,
    state_dim=4,
    lifted_dim=104
).to(device)

anr_model.load_state_dict(
    torch.load(checkpoint_path, map_location=device)
)

anr_model.eval()

#original Koopman Rollout
import torch
import numpy as np
import matplotlib.pyplot as plt

from train_anr_zero_goal import (
    CoNN,
    load_frozen_koopman,
    koopman_step_and_control_jacobian,
)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

koopman_model, device = load_frozen_koopman()
koopman_model.eval()

anr_model = CoNN(30, state_dim=4, lifted_dim=104).to(device)

checkpoint_path = (
    "/Users/havenahn/Project_ANR/"
    "Adjoint-based-Neural-Regulator/ANR/model/"
    "anr_N30_h50_seed_0_e50.pth"
)

anr_model.load_state_dict(
    torch.load(checkpoint_path, map_location=device)
)
anr_model.eval()

# Cost matrix: same as training
R = torch.eye(2, device=device)

def rollout_anr(x0):
    x0 = torch.tensor(
        x0, dtype=torch.float32, device=device
    ).reshape(1, 4)

    with torch.no_grad():
        costates = anr_model(x0)

        eta = koopman_model.enc(x0)
        xi = torch.cat([x0, eta], dim=1)

        states = [x0.cpu().numpy()[0]]
        controls = []

        for k in range(30):
            lambda_next = costates[:, k + 1, :]

            zero_u = torch.zeros(1, 2, device=device)

            _, G = koopman_step_and_control_jacobian(
                koopman_model, xi, zero_u
            )

            G_T_lambda = torch.bmm(
                G.transpose(1, 2),
                lambda_next.unsqueeze(-1)
            ).squeeze(-1)

            u = -0.5 * torch.linalg.solve(
                R, G_T_lambda.unsqueeze(-1)
            ).squeeze(-1)

            xi, _ = koopman_step_and_control_jacobian(
                koopman_model, xi, u
            )

            x = xi[:, :4]

            states.append(x.cpu().numpy()[0])
            controls.append(u.cpu().numpy()[0])

        return np.array(states), np.array(controls)


def rollout_zero_control(x0):
    x0 = torch.tensor(
        x0, dtype=torch.float32, device=device
    ).reshape(1, 4)

    with torch.no_grad():
        eta = koopman_model.enc(x0)
        xi = torch.cat([x0, eta], dim=1)

        states = [x0.cpu().numpy()[0]]

        for k in range(30):
            u = torch.zeros(1, 2, device=device)

            xi, _ = koopman_step_and_control_jacobian(
                koopman_model, xi, u
            )

            states.append(xi[:, :4].cpu().numpy()[0])

        return np.array(states)

#plot trajectories
# Example initial state -- replace with a state
# from the Circle Slider validation dataset.
x0 = [0.5, 0.5, 0.2, 0.1]

states_anr, controls_anr = rollout_anr(x0)
states_zero = rollout_zero_control(x0)

# Position trajectory
plt.figure(figsize=(7, 6))
plt.plot(
    states_anr[:, 0], states_anr[:, 1],
    "o-", label="ANR"
)
plt.plot(
    states_zero[:, 0], states_zero[:, 1],
    "--", label="Zero control"
)
plt.scatter(0, 0, marker="*", s=150, label="Goal")
plt.scatter(x0[0], x0[1], marker="s", label="Start")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid()
plt.legend()
plt.title("Circle Slider: Position Trajectory")
plt.savefig("anr_position_trajectory.png", dpi=200)
plt.show()

# Physical state trajectories
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

labels = ["x", "y", "theta", "v"]

for j, ax in enumerate(axes.flat):
    ax.plot(states_anr[:, j], label="ANR")
    ax.plot(states_zero[:, j], "--", label="Zero control")
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_title(labels[j])
    ax.set_xlabel("Time step")
    ax.grid()
    ax.legend()

plt.tight_layout()
plt.savefig("anr_state_trajectories.png", dpi=200)
plt.show()

# Control trajectories
fig, axes = plt.subplots(2, 1, figsize=(9, 6))

axes[0].plot(controls_anr[:, 0])
axes[0].set_title("Control input u1")
axes[1].plot(controls_anr[:, 1])
axes[1].set_title("Control input u2")

for ax in axes:
    ax.set_xlabel("Time step")
    ax.grid()

plt.tight_layout()
plt.savefig("anr_controls.png", dpi=200)
plt.show()

print("Initial state norm:", np.linalg.norm(states_anr[0]))
print("ANR terminal state norm:", np.linalg.norm(states_anr[-1]))
print("Zero-control terminal state norm:", np.linalg.norm(states_zero[-1]))
print("Maximum absolute control:", np.abs(controls_anr).max(axis=0))
print("Control effort:", np.sum(controls_anr**2))
print("Any NaNs:", np.isnan(states_anr).any() or np.isnan(controls_anr).any())



import torch

def check_tensor(name, t):
    finite = torch.isfinite(t)
    print(
        f"{name}: shape={tuple(t.shape)}, "
        f"finite={finite.all().item()}, "
        f"nan={torch.isnan(t).sum().item()}, "
        f"inf={torch.isinf(t).sum().item()}"
    )
    if finite.any():
        print(
            f"  finite range: "
            f"{t[finite].min().item():.6g} to "
            f"{t[finite].max().item():.6g}"
        )

# 1. Check ANR parameters
for name, p in anr_model.named_parameters():
    if not torch.isfinite(p).all():
        print("BAD ANR PARAMETER:", name)

# 2. Check costate predictions
with torch.no_grad():
    costates = anr_model(x0)

check_tensor("ANR costates", costates)

# 3. Check initial lifting
with torch.no_grad():
    eta0 = koopman_model.enc(x0)
    xi = torch.cat([x0, eta0], dim=1)

check_tensor("Initial lifted state", xi)

# 4. Check the zero-control Koopman rollout, one step at a time
zero_u = torch.zeros((x0.shape[0], 2), device=x0.device)

with torch.no_grad():
    for k in range(30):
        xi_next, G = koopman_step_and_control_jacobian(
            koopman_model, xi, zero_u
        )

        print(f"\nStep {k + 1}")
        check_tensor("xi", xi_next)
        check_tensor("G", G)

        if not torch.isfinite(xi_next).all():
            print("Koopman rollout first becomes non-finite at:", k + 1)
            break

        xi = xi_next