import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import beta
from koopman_dynamics import load_koopman_model

BASE_DIR = Path(__file__).resolve().parent

#Step 1: load Koopman bilinear model

PROJECT_ROOT = Path.home() / "Project_ANR"

KOOPMAN_REPO = (
    PROJECT_ROOT
    / "Koopman-for-Dynamic-Planar-Manipulation-1.0.0"
    / "circle_slider"
    / "bilinear"
    / "nn_multistep_loss_1_bilinear_mu035_copy"
)

CHECKPOINT = KOOPMAN_REPO / "model.pt"


def load_frozen_koopman():
    model, device = load_koopman_model(CHECKPOINT)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    return model, device

# Discrete-time bilinear Koopman step:
# xi[k+1] = A(xi[k]) + B(xi[k]) u[k]
def koopman_step_and_control_jacobian(model, xi, u):
    """
    Args:
        xi: lifted state [batch, D_x + D_e]
        u:  control [batch, 2]

    Returns:
        xi_next: next lifted state [batch, D_x + D_e]
        G:       d xi_next / d u = B_bilinear(xi),
                 shape [batch, D_x + D_e, 2]
    """
    lifted_dim = model.D_x + model.D_e
    G = model.B_bilinear(xi).reshape(-1, lifted_dim, 2)
    xi_next = model.A(xi) + torch.bmm(G, u.unsqueeze(-1)).squeeze(-1)

    A0 = model.A(xi)
    G0 = model.B_bilinear(xi).reshape(-1, lifted_dim, 2)

    Bu0 = torch.bmm(G0, u.unsqueeze(-1)).squeeze(-1)

    # Separate contribution from each control channel
    Bu0_1 = G0[:, :, 0] * u[:, 0:1]
    Bu0_2 = G0[:, :, 1] * u[:, 1:2]

    return xi_next, G


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Step 2: Create Dataset Class
class InitialStateDataset(Dataset):
    def __init__(self, initial_states):
        # Store the initial states as a PyTorch tensor
        self.initial_states = torch.tensor(initial_states, dtype=torch.float32)

    def __len__(self):
        return len(self.initial_states)

    def __getitem__(self, idx):
        # Return a single state
        return self.initial_states[idx]

# Neural Network Model

class CoNN(nn.Module):
    def __init__(self, prediction_horizon, state_dim=4, lifted_dim=104):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.state_dim = state_dim
        self.lifted_dim = lifted_dim

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 128)

        # Predict lambda_1, ..., lambda_N
        self.fc4 = nn.Linear(
            128, (prediction_horizon + 1) * lifted_dim
        )
        

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x.view(
            -1, self.prediction_horizon + 1, self.lifted_dim
        )


# Training Setup
def train_network(
    initial_states,
    n,
    h,
    q1,
    q2,
    q3,
    q4,
    r1,
    r2,
    model_save_path,
    batch_size=128,
    epochs=50,
    lr=2e-4,
):
    dataset = InitialStateDataset(initial_states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load and freeze the pretrained Koopman model.
    koopman_model, device = load_frozen_koopman()
    koopman_model = koopman_model.to(device)
    koopman_model.eval()

    physical_dim = koopman_model.D_x
    lifted_dim = koopman_model.D_x + koopman_model.D_e

    if physical_dim != 4:
        raise ValueError(
            f"This ANR adaptation currently expects 4 physical state "
            f"variables, but the Koopman model has D_x={physical_dim}."
        )

    # CoNN predicts the lifted costate sequence lambda_0,...,lambda_N.
    model = CoNN(
        n,
        state_dim=physical_dim,
        lifted_dim=lifted_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # State and control weights. The reference/goal is the origin:
    # x_ref = [0, 0, 0, 0]^T, matching the zero-reference ANR objective.
    Q = torch.diag(torch.tensor([q1, q2, q3, q4], dtype=torch.float32, device=device))
    R = torch.diag(torch.tensor([r1, r2], dtype=torch.float32, device=device))
    H = h * Q

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_lambda_loss = 0.0

        # for state_0 in dataloader:
        #     optimizer.zero_grad()
        #     state_0 = state_0.to(device=device, dtype=torch.float32)

        #     # Lift the initial physical state. This initializes xi_k,
        #     # which was missing in the previous version of the script.
        #     eta_0 = koopman_model.enc(state_0)
        #     xi_k = torch.cat((state_0, eta_0), dim=1)
        #     state_k = xi_k[:, :physical_dim]

        def check_finite(name, x):
            if not torch.isfinite(x).all():
                print(f"\n[NON-FINITE] {name}")
                print("shape:", x.shape)
                print("NaN:", torch.isnan(x).sum().item())
                print("Inf:", torch.isinf(x).sum().item())
                raise RuntimeError(f"Non-finite value: {name}")

            print(
                f"[OK] {name}: "
                f"max_abs={x.detach().abs().max().item():.6e}"
        )
       
        for state_0 in dataloader:
            optimizer.zero_grad()
            #initial state and costate prediction
            state_0 = state_0.to(device=device, dtype=torch.float32)

            check_finite("state_0", state_0)

            eta_0 = koopman_model.enc(state_0)
            check_finite("eta_0", eta_0)

            xi_k = torch.cat((state_0, eta_0), dim=1)
            check_finite("xi_0", xi_k)

            # --------------------------------
            # Zero-control diagnostic
            # --------------------------------
            with torch.no_grad():
                xi_test = xi_k.clone()

                print("\n--- Zero-control rollout ---")

                for test_step in range(n):
                    zero_u = torch.zeros(
                        xi_test.shape[0], 2,
                        dtype=xi_test.dtype,
                        device=xi_test.device
                    )

                    xi_test, _ = koopman_step_and_control_jacobian(
                        koopman_model, xi_test, zero_u
                    )

                    state_test = xi_test[:, :physical_dim]

                    print(
                        f"Zero-control step {test_step + 1}:",
                        state_test.abs().max().item()
                    )

            state_k = xi_k[:, :physical_dim]

            costate_traj = model(state_0)
            check_finite("costate_traj", costate_traj)

            L_stage = torch.zeros((), device=device)
            lambda_cost = torch.zeros((), device=device)
            #rollout
            for i in range(n):
                #calculate control
                lambda_next = costate_traj[:, i + 1, :]
                check_finite(f"lambda_{i+1}", lambda_next)

                zero_u = torch.zeros(
                    state_0.shape[0], 2,
                    dtype=state_0.dtype,
                    device=device
                )

                _, G = koopman_step_and_control_jacobian(
                    koopman_model, xi_k, zero_u
                )
                check_finite(f"G_{i}", G)

                #calc stae gost
                G_T_lambda = torch.bmm(
                    G.transpose(1, 2),
                    lambda_next.unsqueeze(-1),
                ).squeeze(-1)
                check_finite(f"G_T_lambda_{i}", G_T_lambda)
                # Debug
                print(f"\n--- Step {i} ---")
                print("G max:", G.detach().abs().max().item())
                print(
                    "G per control:",
                    G.detach().abs().amax(dim=(0, 1)).cpu().numpy()
                )
                print("G_T_lambda max:", G_T_lambda.detach().abs().max().item())
                #koopman
                u_opt = -0.5 * torch.linalg.solve(
                    R, G_T_lambda.unsqueeze(-1)
                ).squeeze(-1)
                check_finite(f"u_opt_{i}", u_opt)
                print("u_opt:", u_opt[0].detach().cpu().numpy())

                print("G physical rows:")
                print(G[0, :4, :].detach().cpu().numpy())

                print("G physical rows * u:")
                print(
                    (G[0, :4, :] @ u_opt[0].unsqueeze(-1))
                    .squeeze(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )

                state_error = state_k

                state_cost = torch.einsum(
                    "bi,ij,bj->b", state_error, Q, state_error
                )
                control_cost = torch.einsum(
                    "bi,ij,bj->b", u_opt, R, u_opt
                )

                L_stage = L_stage + (state_cost + control_cost).mean()
                check_finite(f"L_stage_{i}", L_stage)

                lambda_cost = lambda_cost + (
                    costate_traj[:, i, :].abs().sum(dim=1).mean()
                )

                A_k = koopman_model.A(xi_k)
                Bu_k = torch.bmm(
                    G, u_opt.unsqueeze(-1)
                ).squeeze(-1)

                xi_next = A_k + Bu_k

                if i == 0:
                    next_phys = xi_next[:, :physical_dim]
                    abs_next = next_phys.abs()

                    # Find the sample and state dimension with the largest value
                    flat_idx = abs_next.argmax().item()
                    batch_idx = flat_idx // physical_dim
                    dim_idx = flat_idx % physical_dim

                    print("\n--- Step 0: Worst sample ---")
                    print("Batch index:", batch_idx)
                    print("State dimension:", dim_idx)
                    print("Initial state:", state_0[batch_idx].detach().cpu().numpy())
                    print("Control:", u_opt[batch_idx].detach().cpu().numpy())
                    print("A physical:", A_k[batch_idx, :physical_dim].detach().cpu().numpy())
                    print("B u physical:", Bu_k[batch_idx, :physical_dim].detach().cpu().numpy())
                    print("Next physical:", next_phys[batch_idx].detach().cpu().numpy())

                    print(
                        "Max per physical dimension:",
                        abs_next.amax(dim=0).detach().cpu().numpy()
                    )

                xi_k = xi_next
                check_finite(f"xi_{i+1}", xi_k)

                state_k = xi_k[:, :physical_dim]
                check_finite(f"state_{i+1}", state_k)

                terminal_error = state_k

            L_terminal = torch.einsum(
                "bi,ij,bj->b", terminal_error, H, terminal_error
            ).mean()
            check_finite("L_terminal", L_terminal)

            check_finite("lambda_cost", lambda_cost)

            loss = L_stage + L_terminal + beta * lambda_cost
            check_finite("total_loss", loss)

            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    check_finite(f"grad_{name}", param.grad)

            optimizer.step()

            for name, param in model.named_parameters():
                check_finite(f"param_{name}", param)

        avg_loss = epoch_loss / len(dataloader)
        avg_lambda_loss = epoch_lambda_loss / len(dataloader)
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {avg_loss:.6f}, "
            f"Lambda Loss: {avg_lambda_loss:.6f}"
        )

    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path.resolve()}")


if __name__ == '__main__':
    seed = 0
    set_seed(seed)

    n = 30 # Prediction horizon
    h = 50  # Terminal cost coefficient
    epoch = 50
    model_dir = BASE_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Sample initial states from the pretrained Koopman training data

    DATA_PATH = KOOPMAN_REPO.parent / "data" / "Fx_5_50001_0_file0.npy"
    data = np.load(DATA_PATH)

    # Match the preprocessing used to train the Koopman model
    controlled_data = data[:, :3, 2:]

    # First timestep, first 4 channels = physical state
    all_initial_states = controlled_data[:, 0, :4]

    # Sample 10,000 initial states without replacement
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(
        len(all_initial_states),
        size=10000,
        replace=False
    )

    initial_states = all_initial_states[sample_indices].astype(np.float32)

    np.random.shuffle(initial_states)

    print("Initial states shape:", initial_states.shape)
    print("Initial state min:", initial_states.min(axis=0))
    print("Initial state max:", initial_states.max(axis=0))
    
    # Train the model
    q1 = 10.0; q2 = 10.0; q3 = 10.0; q4=10.0; r1 = 1.0; r2 = 1.0
    model_save_path = model_dir / f"anr_N{n}_h{h}_seed_{seed}_e{epoch}.pth"
    train_network(initial_states, n, h, q1, q2, q3,q4, r1, r2, model_save_path, epochs=1, lr=1e-3)
