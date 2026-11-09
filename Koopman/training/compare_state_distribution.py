
import numpy as np

TRAIN_FILE = "../data/Fx_5_50001_0_file0.npy"
ZERO_FILE = "../data/Fx_31_2000_123_zero_control.npy"

NAMES = ["pbody1", "pbody2", "x_dot", "y_dot"]

# Load data
train_raw = np.load(TRAIN_FILE)
zero_raw = np.load(ZERO_FILE)

# Training used only the first 3 timesteps
train_x = train_raw[:, :3, 2:6].reshape(-1, 4)

# All 31 timesteps of zero-control data
zero_x = zero_raw[:, :, 2:6]

print("Training data:", train_raw.shape)
print("Zero-control data:", zero_raw.shape)

# Training distribution
train_min = train_x.min(axis=0)
train_max = train_x.max(axis=0)
train_mean = train_x.mean(axis=0)
train_std = train_x.std(axis=0)

print("\n=== Training distribution (timesteps 0-2) ===")
for i, name in enumerate(NAMES):
    print(
        f"{name:8s} | "
        f"min={train_min[i]:9.4f} "
        f"max={train_max[i]:9.4f} "
        f"mean={train_mean[i]:9.4f} "
        f"std={train_std[i]:9.4f}"
    )

# Zero-control distribution at each timestep
print("\n=== Zero-control distribution by timestep ===")
print(
    "step | "
    + " | ".join(
        f"{name:>12s}" for name in NAMES
    )
    + " | outside training range (%)"
)

for k in range(zero_x.shape[1]):
    x = zero_x[:, k, :]

    outside = (
        (x < train_min) | (x > train_max)
    ).mean(axis=0) * 100

    ranges = []
    for i in range(4):
        ranges.append(
            f"{x[:, i].mean():6.3f}"
        )

    print(
        f"{k:4d} | "
        + " | ".join(ranges)
        + " | "
        + " / ".join(f"{v:6.2f}" for v in outside)
    )

# Additional summary: range and standard deviation
print("\n=== Zero-control range by timestep ===")
print("step | variable | min | max | std")

for k in range(zero_x.shape[1]):
    x = zero_x[:, k, :]
    for i, name in enumerate(NAMES):
        print(
            f"{k:4d} | {name:8s} | "
            f"{x[:, i].min():9.4f} | "
            f"{x[:, i].max():9.4f} | "
            f"{x[:, i].std():9.4f}"
        )
