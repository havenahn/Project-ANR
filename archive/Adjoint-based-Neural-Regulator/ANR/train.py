import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchdiffeq import odeint # Use odeint for integration
import numpy as np
from utils import UnicycleDynamics, set_seed
from config import dt, beta

BASE_DIR = Path(__file__).resolve().parent


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
    def __init__(self, prediction_horizon):
        super(CoNN, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 4*prediction_horizon)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, self.prediction_horizon, 4)
        return x


# Training Setup
def train_network(initial_states, n, h, q1, q2, q3,q4, r1, r2, model_save_path, batch_size=128,epochs=50, lr=2e-4):

    dataset = InitialStateDataset(initial_states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')                # Force CPU usage for training
    print("Using device:", device)
    # Time span for integration
    t_span = torch.tensor([0, dt], dtype=torch.float32, device=device)
    ode_solver = UnicycleDynamics()

    # Initialize NN
    model = CoNN(n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    # Define cost matrices
    Q = torch.diag(torch.tensor([q1, q2, q3, q4], device=device))  # State cost
    R = torch.diag(torch.tensor([r1, r2], device=device))      # Control input cost
    H =  h*Q                                                   # Terminal cost

    for epoch in range(epochs):     
        epoch_loss = 0
        epoch_bc_loss = 0
        epoch_lambda_loss = 0
        # Iterate over all initial conditions
        for state_0 in dataloader:

            optimizer.zero_grad()
            state_0 = state_0.to(device)

            costate_traj_k_hat = model(state_0)   # Predicted co-state trajectory starting at time k
            state_k = state_0
            L_stage = 0
            L_terminal = 0
            lambda_cost = 0
            
            
            for i in range(n):
                lambdas_i = costate_traj_k_hat[:, i, :]   # [B, 4]
                lambda1_i = lambdas_i[:, 0]
                lambda2_i = lambdas_i[:, 1]
                lambda3_i = lambdas_i[:, 2]
                lambda4_i = lambdas_i[:, 3]
                theta_i = state_k[:, 2]                   # [B]

                a_opt = -0.5 * lambda4_i /r1              # [B]
                w_opt = -0.5 * lambda3_i/r2             # [B]
                u_opt = torch.stack([a_opt, w_opt], dim=1)  # [B, 2]


                # Compute stage cost using matrices
                state_cost = torch.einsum('bi,ij,bj->b', state_k, Q, state_k)   # [B]
                control_cost = torch.einsum('bi,ij,bj->b', u_opt, R, u_opt)     # [B]


                lambda_cost += lambdas_i.abs().sum(dim=1).mean()
                L_stage += (state_cost + control_cost).mean()

                # Solve the initial value problem using odeint (Step simulation forward by dt)
                x_and_u = torch.cat([state_k, u_opt], dim=1).to(device)
                result = odeint(ode_solver, x_and_u, t_span, method='rk4')
                state_k = result[-1, :, :4]
                theta_wrapped = torch.atan2(torch.sin(state_k[:, 2]), torch.cos(state_k[:, 2]))
                state_k = torch.cat([state_k[:, :2], theta_wrapped.unsqueeze(1), state_k[:, 3:4]], dim=1)


            # Compute L_terminal
            L_terminal = torch.einsum('bi,ij,bj->b', state_k, H, state_k).mean() 
    
            # Backpropagation
            loss = L_stage + L_terminal + (beta*lambda_cost) 
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_lambda_loss += lambda_cost.item()
            

        avg_loss = epoch_loss / len(dataloader)
        avg_lambda_loss = epoch_lambda_loss / len(dataloader)
        print(f"********Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.2f}, Lambda Loss: {avg_lambda_loss:.2f}********")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {Path(model_save_path).resolve()}")

if __name__ == '__main__':
    seed = 0
    set_seed(seed)

    n = 30 # Prediction horizon
    h = 50  # Terminal cost coefficient
    epoch = 50
    model_dir = BASE_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate  combinations of (x, y, theta)
    x_range = np.linspace(-2, 2, 10)
    y_range = np.linspace(-2, 2, 10)
    theta_range = np.linspace(-2, 2, 10) 
    v_range = np.linspace(-2, 2, 10) 

    # Create a grid of all combinations
    x, y, theta, v = np.meshgrid(x_range, y_range, theta_range, v_range)
    initial_states = np.vstack([x.ravel(), y.ravel(), theta.ravel(), v.ravel()]).T

    # Randomly shuffle the data set
    np.random.shuffle(initial_states)
    
    # Train the model
    q1 = 10.0; q2 = 10.0; q3 = 10.0; q4=10.0; r1 = 1.0; r2 = 1.0
    model_save_path = model_dir / f"anr_N{n}_h{h}_seed_{seed}_e{epoch}.pth"
    train_network(initial_states, n, h, q1, q2, q3,q4, r1, r2, model_save_path, epochs=epoch, lr=1e-3)
