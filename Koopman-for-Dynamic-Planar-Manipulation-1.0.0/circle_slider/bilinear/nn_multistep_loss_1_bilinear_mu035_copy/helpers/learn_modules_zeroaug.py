#!/usr/bin/env python3
"""Fine-tune only Koopman's A matrix using controlled + physical zero-control data.

The encoder and B_bilinear network remain exactly frozen from the pretrained
checkpoint. No spectral-norm or eigenvalue clipping is applied.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from helpers.networkarch import NeuralNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


class L3ZeroControlFineTune:
    def __init__(
        self,
        initial_checkpoint: str = "model.pt",
        output_checkpoint: str = "model_zeroaug.pt",
        n_x: int = 4,
        n_h: int = 32,
        n_e: int = 100,
        epochs: int = 50,
        controlled_batch_size: int = 8192,
        zero_batch_size: int = 128,
        learning_rate: float = 1e-5,
        zero_loss_weight: float = 2.0,
        gradient_clip: float = 1.0,
        seed: int = 42,
    ):
        self.initial_checkpoint = Path(initial_checkpoint)
        self.output_checkpoint = Path(output_checkpoint)
        self.n_x = n_x
        self.n_h = n_h
        self.n_e = n_e
        self.epochs = epochs
        self.controlled_batch_size = controlled_batch_size
        self.zero_batch_size = zero_batch_size
        self.learning_rate = learning_rate
        self.zero_loss_weight = zero_loss_weight
        self.gradient_clip = gradient_clip
        self.seed = seed
        self.model: NeuralNetwork | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    @staticmethod
    def _load_state_dict(checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")
        checkpoint = torch.load(
            checkpoint_path, map_location=DEVICE, weights_only=False
        )
        return checkpoint.get("model_dict", checkpoint)

    def _build_model(self):
        model = NeuralNetwork(N_x=self.n_x, N_h=self.n_h, N_e=self.n_e).to(DEVICE)
        state_dict = self._load_state_dict(self.initial_checkpoint)
        model.load_state_dict(state_dict, strict=True)

        # Preserve the pretrained encoder and control map exactly.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.A.weight.requires_grad_(True)
        model.eval()
        return model

    def _step(self, batch: torch.Tensor, normalize_over_horizon: bool):
        # DataLoader over TensorDataset returns a one-element tuple/list.
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        batch = batch.to(device=DEVICE, dtype=DTYPE)
        states = batch[:, :, : self.n_x]
        controls = batch[:, :, self.n_x :]

        x_hat, eta_hat = self.model(states[:, 0, :], controls[:, 0, :])
        state_losses = [torch.nn.functional.mse_loss(x_hat, states[:, 1, :])]
        eta_losses = [
            torch.nn.functional.mse_loss(
                eta_hat, self.model.enc(states[:, 1, :])
            )
        ]

        xi_hat = torch.cat((x_hat, eta_hat), dim=1)
        time_weight = 0.95
        weights = [1.0]
        for time_step in range(states.shape[1] - 2):
            x_hat, eta_hat = self.model.ldm(
                xi_hat, controls[:, time_step + 1, :]
            )
            xi_hat = torch.cat((x_hat, eta_hat), dim=1)
            weight = time_weight ** (time_step + 1)
            weights.append(weight)
            state_losses.append(
                weight
                * torch.nn.functional.mse_loss(
                    x_hat, states[:, time_step + 2, :]
                )
            )
            eta_losses.append(
                weight
                * torch.nn.functional.mse_loss(
                    eta_hat,
                    self.model.enc(states[:, time_step + 2, :]),
                )
            )

        state_loss = torch.stack(state_losses).sum()
        eta_loss = torch.stack(eta_losses).sum()

        if normalize_over_horizon:
            normalizer = float(sum(weights))
            state_loss = state_loss / normalizer
            eta_loss = eta_loss / normalizer

        total_loss = state_loss + 100.0 * eta_loss
        return total_loss, state_loss

    @staticmethod
    def _split(data: torch.Tensor, seed: int):
        n_train = int(0.6 * len(data))
        n_val = len(data) - n_train
        generator = torch.Generator().manual_seed(seed)
        return random_split(
            TensorDataset(data), [n_train, n_val], generator=generator
        )

    @staticmethod
    def _mean_loss(loader, model_step, normalize: bool):
        losses, state_losses = [], []
        with torch.no_grad():
            for batch in loader:
                loss, state_loss = model_step(batch, normalize)
                losses.append(float(loss.item()))
                state_losses.append(float(state_loss.item()))
        if not losses:
            raise RuntimeError("A validation split is empty")
        return float(np.mean(losses)), float(np.mean(state_losses))

    def _evaluate_zero_rollout(self, loader, label: str):
        """Report held-out physical-state RMSE and max magnitude by horizon step."""
        squared_error = None
        element_count = 0
        max_abs_by_step = None
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                trajectories = batch[0].to(device=DEVICE, dtype=DTYPE)
                states = trajectories[:, :, : self.n_x]
                controls = trajectories[:, :, self.n_x :]
                xi = torch.cat((states[:, 0, :], self.model.enc(states[:, 0, :])), dim=1)

                if squared_error is None:
                    horizon = states.shape[1] - 1
                    squared_error = torch.zeros(horizon, device=DEVICE)
                    max_abs_by_step = torch.zeros(horizon, device=DEVICE)

                for k in range(states.shape[1] - 1):
                    x_next, eta_next = self.model.ldm(xi, controls[:, k, :])
                    xi = torch.cat((x_next, eta_next), dim=1)
                    error = x_next - states[:, k + 1, :]
                    squared_error[k] += error.square().sum()
                    element_count += error.numel()
                    max_abs_by_step[k] = torch.maximum(
                        max_abs_by_step[k], x_next.abs().max()
                    )
                    if not torch.isfinite(xi).all():
                        raise RuntimeError(
                            f"Non-finite zero-control prediction at step {k + 1} ({label})"
                        )

        if squared_error is None:
            raise RuntimeError("Zero-control validation loader is empty")

        # Each horizon step has the same number of physical-state elements.
        n_samples = element_count // (len(squared_error) * self.n_x)
        rmse_by_step = torch.sqrt(squared_error / (n_samples * self.n_x))
        print(f"\n{label} held-out zero-control rollout:")
        for k in range(len(squared_error)):
            if (k + 1) % 5 == 0 or k == len(squared_error) - 1:
                print(
                    f"  step {k + 1:02d}: RMSE={rmse_by_step[k].item():.6e}, "
                    f"max_abs_state={max_abs_by_step[k].item():.6e}"
                )

    def fit(self, controlled_data: np.ndarray, zero_control_data: np.ndarray):
        controlled = torch.as_tensor(controlled_data, dtype=DTYPE)
        zero = torch.as_tensor(zero_control_data, dtype=DTYPE)

        if controlled.ndim != 3 or zero.ndim != 3:
            raise ValueError("Both datasets must have shape [trajectories, time, 6]")
        if controlled.shape[2] != self.n_x + 2 or zero.shape[2] != self.n_x + 2:
            raise ValueError("Expected 4 state channels followed by 2 control channels")
        if controlled.shape[1] < 3 or zero.shape[1] < 3:
            raise ValueError("Each trajectory must contain at least 3 time points")
        if not torch.isfinite(controlled).all() or not torch.isfinite(zero).all():
            raise ValueError("Training data contains NaN or Inf")

        self.model = self._build_model()
        original_A = self.model.A.weight.detach().clone()
        original_eig_radius = float(
            torch.linalg.eigvals(self.model.A.weight.detach()).abs().max().item()
        )
        original_norm = float(
            torch.linalg.matrix_norm(self.model.A.weight.detach(), ord=2).item()
        )

        self.optimizer = torch.optim.Adam(
            [self.model.A.weight], lr=self.learning_rate
        )

        controlled_train, controlled_val = self._split(controlled, self.seed)
        zero_train, zero_val = self._split(zero, self.seed + 1)

        controlled_train_loader = DataLoader(
            controlled_train,
            batch_size=self.controlled_batch_size,
            shuffle=True,
        )
        zero_train_loader = DataLoader(
            zero_train,
            batch_size=self.zero_batch_size,
            shuffle=True,
        )
        controlled_val_loader = DataLoader(
            controlled_val, batch_size=self.controlled_batch_size, shuffle=False
        )
        zero_val_loader = DataLoader(
            zero_val, batch_size=self.zero_batch_size, shuffle=False
        )

        print(f"Device: {DEVICE}")
        print(f"Controlled data: {tuple(controlled.shape)}")
        print(f"Zero-control data: {tuple(zero.shape)}")
        print("Trainable parameters: model.A.weight only")
        print("Frozen parameters: encoder and B_bilinear")
        print(f"Initial spectral radius(A): {original_eig_radius:.6f}")
        print(f"Initial ||A||_2: {original_norm:.6f}")
        self._evaluate_zero_rollout(zero_val_loader, label="BASELINE")

        for epoch in range(self.epochs):
            self.model.eval()  # architecture has no dropout or batch normalization
            train_losses, train_controlled, train_zero = [], [], []
            zero_iterator = iter(zero_train_loader)

            for controlled_batch in controlled_train_loader:
                try:
                    zero_batch = next(zero_iterator)
                except StopIteration:
                    zero_iterator = iter(zero_train_loader)
                    zero_batch = next(zero_iterator)

                controlled_loss, _ = self._step(
                    controlled_batch, normalize_over_horizon=False
                )
                zero_loss, _ = self._step(
                    zero_batch, normalize_over_horizon=True
                )
                total_loss = controlled_loss + self.zero_loss_weight * zero_loss

                if not torch.isfinite(total_loss):
                    raise RuntimeError(
                        f"Non-finite combined loss at epoch {epoch + 1}"
                    )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [self.model.A.weight], max_norm=self.gradient_clip
                )
                self.optimizer.step()

                train_losses.append(float(total_loss.detach().item()))
                train_controlled.append(float(controlled_loss.detach().item()))
                train_zero.append(float(zero_loss.detach().item()))

            val_controlled, _ = self._mean_loss(
                controlled_val_loader, self._step, normalize=False
            )
            val_zero, _ = self._mean_loss(
                zero_val_loader, self._step, normalize=True
            )
            rho = float(
                torch.linalg.eigvals(self.model.A.weight.detach()).abs().max().item()
            )
            relative_A_change = float(
                (torch.linalg.norm(self.model.A.weight.detach() - original_A)
                 / torch.linalg.norm(original_A)).item()
            )

            print(
                f"Epoch {epoch + 1:03d}/{self.epochs} | "
                f"train total={np.mean(train_losses):.6e} | "
                f"train controlled={np.mean(train_controlled):.6e} | "
                f"train zero={np.mean(train_zero):.6e} | "
                f"val controlled={val_controlled:.6e} | "
                f"val zero={val_zero:.6e} | "
                f"rho(A)={rho:.5f} | "
                f"relative ΔA={relative_A_change:.3%}"
            )

        self.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_dict": self.model.state_dict()}, self.output_checkpoint
        )
        torch.save(
            {"opt_dict": self.optimizer.state_dict()},
            self.output_checkpoint.with_name(
                self.output_checkpoint.stem + "_optimizer.pt"
            ),
        )

        final_A = self.model.A.weight.detach()
        final_rho = float(torch.linalg.eigvals(final_A).abs().max().item())
        final_norm = float(torch.linalg.matrix_norm(final_A, ord=2).item())
        final_change = float(
            (torch.linalg.norm(final_A - original_A) / torch.linalg.norm(original_A)).item()
        )
        print("\nSaved:", self.output_checkpoint.resolve())
        print(f"Final spectral radius(A): {final_rho:.6f}")
        print(f"Final ||A||_2: {final_norm:.6f}")
        print(f"Relative A change: {final_change:.3%}")
        print("Encoder and B_bilinear are unchanged from the original checkpoint.")
        self._evaluate_zero_rollout(zero_val_loader, label="FINE-TUNED")
        return self.model
