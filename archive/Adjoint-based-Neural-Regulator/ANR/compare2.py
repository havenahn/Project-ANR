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